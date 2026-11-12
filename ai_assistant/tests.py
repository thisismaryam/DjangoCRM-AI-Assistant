import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.contrib.auth.models import Permission
from django.db import IntegrityError
from django.test import TestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from rest_framework.test import APIRequestFactory, force_authenticate

from ai_assistant.api_views import approve_view
from ai_assistant.agent import agent_node
from ai_assistant.context import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_TURNS,
    trim_conversation_history,
)
from ai_assistant.llm import build_fallback_llm
from common.models import Department
from crm.models import City, ClientType, Company, Country
from ai_assistant.tools import (
    MAX_SCHEMA_RESULT_CHARS,
    list_departments,
    prepare_company,
    prepare_lead,
    search_database_schema,
    tools,
)


class TestAssistantTools(TestCase):
    def setUp(self):
        Group.objects.get_or_create(name="co-workers")

    @patch("ai_assistant.tools.SCHEMA_CATALOG")
    @patch(
        "ai_assistant.tools.search_schema",
        return_value=[{"model": "Lead"}, {"model": "Deal"}],
    )
    def test_schema_search_returns_one_model_and_at_most_ten_field_names_and_types(
        self, search_schema, schema_catalog
    ):
        schema_catalog.__getitem__.return_value = {
            "description": "Should not be returned",
            "fields": [
                {"name": f"field_{index}", "type": "CharField", "required": True}
                for index in range(12)
            ],
        }

        result = search_database_schema.invoke({"query": "lead contact"})

        self.assertEqual(
            result,
            "Model: Lead\nFields:\n"
            + "\n".join(f"- field_{index}: CharField" for index in range(10)),
        )
        search_schema.assert_called_once_with("lead contact", top_k=1)

    @patch("ai_assistant.tools.SCHEMA_CATALOG")
    @patch("ai_assistant.tools.search_schema", return_value=[{"model": "Lead"}])
    def test_schema_search_respects_hard_result_size_limit(
        self, search_schema, schema_catalog
    ):
        schema_catalog.__getitem__.return_value = {
            "fields": [
                {"name": "x" * 2000, "type": "CharField", "required": True}
            ]
        }

        result = search_database_schema.invoke({"query": "lead"})

        self.assertLessEqual(len(result), MAX_SCHEMA_RESULT_CHARS)
        self.assertNotIn("required", result)
        search_schema.assert_called_once_with("lead", top_k=1)

    def test_context_trimming_keeps_recent_complete_tool_turns(self):
        messages = [
            HumanMessage(content="old " * MAX_HISTORY_CHARS),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_database_schema",
                    "args": {"query": "recent question"},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            ),
            ToolMessage(content="Model: Lead\nFields:\n- name: CharField", tool_call_id="call-1"),
        ]

        trimmed = trim_conversation_history(messages)

        self.assertEqual(len(trimmed), 3)
        self.assertIsInstance(trimmed[0], HumanMessage)
        self.assertIsInstance(trimmed[1], AIMessage)
        self.assertIsInstance(trimmed[2], ToolMessage)

    def test_context_trimming_retains_latest_two_user_turns_for_follow_up(self):
        messages = [
            HumanMessage(content="An older unrelated request"),
            AIMessage(content="An older answer"),
            HumanMessage(content="Create Northstar Labs in California."),
            AIMessage(content="I have Northstar Labs in California. Which city?"),
            HumanMessage(content="San Francisco"),
        ]

        trimmed = trim_conversation_history(messages)

        self.assertEqual(MAX_HISTORY_TURNS, 2)
        self.assertEqual(trimmed, messages[2:])

    def test_agent_prompt_merges_brief_follow_up_with_recent_request(self):
        messages = [
            HumanMessage(content="Create Northstar Labs in California."),
            AIMessage(content="I have Northstar Labs in California. Which city?"),
            HumanMessage(content="San Francisco"),
        ]

        with patch("ai_assistant.agent.llm", autospec=True) as llm:
            llm.invoke.return_value = AIMessage(
                content="I will prepare Northstar Labs."
            )
            agent_node({"messages": messages})

        prompt = llm.invoke.call_args.args[0]
        self.assertEqual(prompt[1:], messages)
        self.assertIn("Treat brief user replies as answers", prompt[0].content)
        self.assertIn("Do NOT ask for an email", prompt[0].content)
        self.assertIn(
            "Call it as soon as the company name is known",
            prompt[0].content,
        )
        self.assertIn("Do NOT ask for an email", prompt[0].content)
        self.assertIn(
            "Do NOT ask the user to confirm details before calling prepare_company",
            prompt[0].content,
        )
        self.assertIn(
            "repeat the tool's error message verbatim",
            prompt[0].content,
        )

    def test_approved_lead_result_contains_crm_deep_link(self):
        lead = type("LeadResult", (), {"id": 42, "first_name": "Ada"})()
        with (
            patch("ai_assistant.tools.interrupt", return_value="approved"),
            patch(
                "ai_assistant.tools._current_user",
                return_value=SimpleNamespace(has_perm=lambda _permission: True),
            ),
            patch("ai_assistant.tools.get_user_model") as get_user_model,
            patch("ai_assistant.tools.Lead.objects.filter") as filter_leads,
            patch("ai_assistant.tools.Lead.objects.create", return_value=lead),
        ):
            filter_leads.return_value.exists.return_value = False
            user_model = get_user_model.return_value
            user_model.objects.filter.return_value.first.return_value = None

            result = prepare_lead.invoke(
                {"first_name": "Ada", "email": "ada@example.com"},
                config={"configurable": {"user_id": 1}},
            )

        self.assertIn("SUCCESS: Lead for Ada created.", result)
        self.assertIn(
            "<a href='/en/456-admin/crm/lead/42/change/' target='_blank'>"
            "Click here to view the Lead in CRM</a>",
            result,
        )

    def test_company_creation_approves_and_creates_missing_related_records(self):
        department = Department.objects.create(name="Company tool department")
        user = get_user_model().objects.create_user(
            username="company-tool-user",
            password="test-password",
        )
        user.groups.add(department)
        permissions = Permission.objects.filter(
            content_type__app_label="crm",
            codename__in=(
                "add_company",
                "add_country",
                "add_city",
                "add_clienttype",
            ),
        )
        user.user_permissions.add(*permissions)

        with patch(
            "ai_assistant.tools.interrupt",
            return_value="approved",
        ) as interrupt:
            result = prepare_company.invoke(
                {
                    "full_name": "Northstar Labs",
                    "country_name": "New Zealand",
                    "city_name": "Wellington",
                    "type_name": "Strategic Partner",
                    "phone": "+64 4 555 0100",
                    "website": "https://northstar.example",
                    "registration_number": "NZ-100",
                },
                config={"configurable": {"user_id": user.pk}},
            )

        self.assertIn("SUCCESS", result, result)
        country = Country.objects.get(name="New Zealand")
        city = City.objects.get(name="Wellington", country=country)
        client_type = ClientType.objects.get(
            name="Strategic Partner",
            department=department,
        )
        company = Company.objects.get(full_name="Northstar Labs")
        self.assertEqual(country.url_name, "new-zealand")
        self.assertEqual(company.country, country)
        self.assertEqual(company.city, city)
        self.assertEqual(company.type, client_type)
        self.assertEqual(company.email, "")
        self.assertEqual(company.phone, "+64 4 555 0100")
        self.assertEqual(company.website, "https://northstar.example")
        self.assertEqual(company.registration_number, "NZ-100")
        self.assertEqual(company.owner, user)
        self.assertEqual(company.department_id, department.pk)
        self.assertEqual(
            interrupt.call_args.args[0]["data"]["related_records_to_create"],
            ["country", "city", "type"],
        )
        self.assertIn("SUCCESS", result)

    def test_company_reuses_named_related_records_case_insensitively(self):
        department = Department.objects.create(name="Existing company department")
        user = get_user_model().objects.create_user(
            username="existing-company-user",
            password="test-password",
        )
        user.groups.add(department)
        user.user_permissions.add(
            *Permission.objects.filter(
                content_type__app_label="crm",
                codename__in=("add_company",),
            )
        )
        country = Country.objects.create(
            name="Iceland",
            url_name="iceland",
        )
        city = City.objects.create(name="Reykjavik", country=country)
        client_type = ClientType.objects.create(
            name="Reseller",
            department=department,
        )

        with patch(
            "ai_assistant.tools.interrupt",
            return_value="approved",
        ) as interrupt:
            prepare_company.invoke(
                {
                    "full_name": "Aurora Traders",
                    "country_name": "iceland",
                    "city_name": "reykjavik",
                    "type_name": "reseller",
                },
                config={"configurable": {"user_id": user.pk}},
            )

        company = Company.objects.get(full_name="Aurora Traders")
        self.assertEqual(company.country, country)
        self.assertEqual(company.city, city)
        self.assertEqual(company.type, client_type)
        self.assertEqual(
            interrupt.call_args.args[0]["data"]["related_records_to_create"],
            [],
        )

    def test_company_rejection_does_not_create_company_or_related_records(self):
        user = get_user_model().objects.create_user(
            username="rejected-company-user",
            password="test-password",
        )
        user.user_permissions.add(
            *Permission.objects.filter(
                content_type__app_label="crm",
                codename__in=(
                    "add_company",
                    "add_country",
                    "add_city",
                ),
            )
        )

        with patch("ai_assistant.tools.interrupt", return_value="rejected"):
            result = prepare_company.invoke(
                {
                    "full_name": "Rejected Company",
                    "country_name": "Lumeria",
                    "city_name": "Aster",
                },
                config={"configurable": {"user_id": user.pk}},
            )

        self.assertIn("CANCELLED", result)
        self.assertFalse(Company.objects.filter(full_name="Rejected Company").exists())
        self.assertFalse(Country.objects.filter(name="Lumeria").exists())
        self.assertFalse(City.objects.filter(name="Aster").exists())

    def test_company_creation_does_not_require_or_store_email(self):
        user = get_user_model().objects.create_user(
            username="email-free-company-user",
            password="test-password",
        )
        user.user_permissions.add(
            *Permission.objects.filter(
                content_type__app_label="crm",
                codename__in=("add_company",),
            )
        )

        with patch("ai_assistant.tools.interrupt", return_value="approved"):
            result = prepare_company.invoke(
                {"full_name": "Email-Free Company"},
                config={"configurable": {"user_id": user.pk}},
            )
        city_without_country = prepare_company.invoke(
            {
                "full_name": "Email-Free Company",
                "city_name": "San Francisco",
            },
            config={"configurable": {"user_id": user.pk}},
        )

        company = Company.objects.get(full_name="Email-Free Company")
        self.assertEqual(company.email, "")
        self.assertIn("SUCCESS", result)
        self.assertIn("NEEDS_DETAILS", city_without_country)
        self.assertIn("country is required", city_without_country)

    def test_company_tool_schema_does_not_include_email(self):
        self.assertNotIn("email", prepare_company.args_schema.model_fields)

    def test_new_company_type_requires_one_user_department(self):
        user = get_user_model().objects.create_user(
            username="departmentless-company-user",
            password="test-password",
        )
        user.user_permissions.add(
            *Permission.objects.filter(
                content_type__app_label="crm",
                codename__in=("add_company", "add_clienttype"),
            )
        )

        with patch("ai_assistant.tools.interrupt") as interrupt:
            result = prepare_company.invoke(
                {
                    "full_name": "No Department Company",
                    "type_name": "Unassigned Type",
                },
                config={"configurable": {"user_id": user.pk}},
            )

        self.assertIn("ERROR:", result)
        self.assertIn("0 departments; exactly one is required", result)
        interrupt.assert_not_called()
        self.assertFalse(
            ClientType.objects.filter(name="Unassigned Type").exists()
        )

    def test_company_type_with_multiple_user_departments_returns_clean_error(self):
        first = Department.objects.create(name="First company department")
        second = Department.objects.create(name="Second company department")
        user = get_user_model().objects.create_user(
            username="multi-department-company-user",
            password="test-password",
        )
        user.groups.add(first, second)
        user.user_permissions.add(
            *Permission.objects.filter(
                content_type__app_label="crm",
                codename__in=("add_company", "add_clienttype"),
            )
        )

        with patch("ai_assistant.tools.interrupt") as interrupt:
            result = prepare_company.invoke(
                {
                    "full_name": "Multiple Department Company",
                    "type_name": "New Company Type",
                },
                config={"configurable": {"user_id": user.pk}},
            )

        self.assertIn("ERROR:", result)
        self.assertIn("2 departments; exactly one is required", result)
        self.assertFalse(
            Company.objects.filter(
                full_name="Multiple Department Company"
            ).exists()
        )
        self.assertFalse(ClientType.objects.filter(name="New Company Type").exists())
        interrupt.assert_not_called()

    def test_list_departments_shows_only_authenticated_users_departments(self):
        department = Department.objects.create(name="Visible department")
        Department.objects.create(name="Other department")
        user = get_user_model().objects.create_user(
            username="department-list-user",
            password="test-password",
        )
        user.groups.add(department)

        result = list_departments.invoke(
            {},
            config={"configurable": {"user_id": user.pk}},
        )

        self.assertEqual(
            json.loads(result),
            [{"id": department.pk, "name": "Visible department"}],
        )


class TestAssistantApi(TestCase):
    @patch(
        "ai_assistant.api_views.app.get_state",
        return_value=SimpleNamespace(values={"messages": ["started"]}),
    )
    @patch(
        "ai_assistant.api_views.app.invoke",
        return_value={
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "action": "CREATE_LEAD",
                        "data": {"first_name": "Ada", "email": "ada@example.com"},
                    }
                )
            ]
        },
    )
    def test_approval_endpoint_returns_pending_lead_approval(
        self, invoke, get_state
    ):
        request = APIRequestFactory().post(
            "/api/ai/approve/",
            {"thread_id": "lead-workflow", "decision": "approved"},
            format="json",
        )
        force_authenticate(
            request,
            user=SimpleNamespace(pk=73, is_authenticated=True, is_active=True),
        )

        response = approve_view(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "pending_approval")
        self.assertEqual(response.data["thread_id"], "lead-workflow")
        self.assertEqual(response.data["payload"]["action"], "CREATE_LEAD")
        invoke.assert_called_once()
        self.assertEqual(
            invoke.call_args.kwargs["config"]["configurable"],
            {"thread_id": "user-73:lead-workflow", "user_id": 73},
        )
        get_state.assert_called_once()

    @patch("ai_assistant.api_views.app.invoke")
    def test_approval_endpoint_requires_authentication(self, invoke):
        request = APIRequestFactory().post(
            "/api/ai/approve/",
            {"thread_id": "someone-elses-thread", "decision": "approved"},
            format="json",
        )

        response = approve_view(request)

        self.assertIn(response.status_code, (401, 403))
        invoke.assert_not_called()

    @patch(
        "ai_assistant.api_views.app.get_state",
        return_value=SimpleNamespace(values={}),
    )
    @patch("ai_assistant.api_views.app.invoke")
    def test_user_cannot_resume_another_users_thread(self, invoke, get_state):
        request = APIRequestFactory().post(
            "/api/ai/approve/",
            {"thread_id": "project-workflow", "decision": "approved"},
            format="json",
        )
        force_authenticate(
            request,
            user=SimpleNamespace(pk=74, is_authenticated=True, is_active=True),
        )

        response = approve_view(request)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["status"], "error")
        self.assertEqual(
            get_state.call_args.args[0]["configurable"]["thread_id"],
            "user-74:project-workflow",
        )
        invoke.assert_not_called()

    @patch(
        "ai_assistant.api_views.app.get_state",
        return_value=SimpleNamespace(values={"messages": ["started"]}),
    )
    @patch(
        "ai_assistant.api_views.app.invoke",
        side_effect=IntegrityError("invalid foreign key"),
    )
    def test_approval_database_errors_return_json_client_error(
        self, invoke, get_state
    ):
        request = APIRequestFactory().post(
            "/api/ai/approve/",
            {"thread_id": "my-thread", "decision": "approved"},
            format="json",
        )
        force_authenticate(
            request,
            user=SimpleNamespace(pk=73, is_authenticated=True, is_active=True),
        )

        response = approve_view(request)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["status"], "error")
        self.assertIn("invalid or unavailable", response.data["message"])

    def test_agent_tool_registry_includes_gemini_and_groq_callables(self):
        self.assertEqual(
            {tool.name for tool in tools},
            {
                "search_database_schema",
                "prepare_lead",
                "prepare_company",
                "list_departments",
            },
        )


class TestLlmFallback(TestCase):
    def test_falls_back_to_tool_capable_groq_runnable(self):
        def fail_gemini(prompt):
            raise RuntimeError(f"Gemini unavailable for: {prompt}")

        primary = RunnableLambda(fail_gemini)
        fallback = RunnableLambda(
            lambda prompt: AIMessage(
                content=f"Fallback response for: {prompt}",
                tool_calls=[
                    {
                        "name": "prepare_lead",
                        "args": {"first_name": "Ada", "email": "ada@example.com"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        )

        response = build_fallback_llm(primary, [fallback]).invoke(
            "Create a lead for Ada."
        )

        self.assertEqual(response.tool_calls[0]["name"], "prepare_lead")
