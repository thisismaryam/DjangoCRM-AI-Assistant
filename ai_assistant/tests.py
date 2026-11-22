import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
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
from ai_assistant.llm import (
    build_fallback_llm,
    gemini_fallback,
    gemini_primary,
)
from common.models import Department
from crm.models import City, Company, Country, Lead
from ai_assistant.tools import (
    MAX_SCHEMA_TOOL_CHARS,
    list_departments,
    prepare_lead,
    search_companies,
    search_contacts,
    search_database_schema,
    tools,
)


class TestAssistantTools(TestCase):
    def setUp(self):
        Group.objects.get_or_create(name="co-workers")

    @patch("ai_assistant.tools.search_schema_with_relationships")
    def test_schema_search_shows_business_fields_not_audit_fields(self, search):
        search.return_value = {
            "matched_models": ["Lead"],
            "expanded_models": ["Company"],
            "schemas": {
                "Lead": {
                    "fields": [
                        {"name": "creation_date", "type": "DateTimeField",
                         "required_for_input": False},
                        {"name": "phone", "type": "CharField",
                         "required_for_input": False},
                        {"name": "email", "type": "CharField",
                         "required_for_input": True},
                        {"name": "company", "verbose_name": "Company of contact",
                         "type": "ForeignKey", "required_for_input": False},
                        {"name": "tags", "type": "ManyToManyField",
                         "required_for_input": False},
                    ],
                    "relationships": [
                        {"field": "company", "related_model": "Company",
                         "type": "ForeignKey"},
                        {"field": "tags", "related_model": "Tag",
                         "type": "ManyToMany"},
                    ],
                    "reverse_relationships": [],
                },
                "Company": {"fields": [], "relationships": [],
                            "reverse_relationships": []},
            },
        }

        result = search_database_schema.invoke({"query": "lead"})

        self.assertNotIn("creation_date", result)
        # required fields come first and are starred
        self.assertTrue(result.split("Fields (* = required): ")[1]
                        .startswith("email: CharField*"))
        self.assertIn("company (Company of contact) -> Company", result)
        self.assertIn("tags -> [Tag]", result)
        self.assertIn("phone: CharField", result)
        self.assertIn("## Company", result)

    def test_schema_formatter_handles_every_model_in_real_catalog(self):
        from ai_assistant.tools import SCHEMA_CATALOG, _format_schema_entry
        self.assertIn("Lead", SCHEMA_CATALOG)
        for name, entry in SCHEMA_CATALOG.items():
            for matched in (True, False):
                self.assertTrue(_format_schema_entry(name, entry, matched))

    def test_real_lead_schema_lists_required_first_and_no_audit_fields(self):
        from ai_assistant.tools import SCHEMA_CATALOG, _format_schema_entry
        text = _format_schema_entry("Lead", SCHEMA_CATALOG["Lead"], True)
        fields = text.split("Fields (* = required): ")[1].split("\n")[0]
        self.assertTrue(fields.startswith("first_name: CharField*, email: CharField*"))
        for hidden in ("creation_date", "update_date", "modified_by", "token"):
            self.assertNotIn(hidden, fields)
        for wanted in ("lead_source -> LeadSource", "company_name", "website",
                       "country -> Country", "tags -> [Tag]"):
            self.assertIn(wanted, fields)

    @patch("ai_assistant.tools.search_schema_with_relationships")
    def test_schema_search_respects_hard_result_size_limit(self, search):
        schemas = {
            f"Model{i}": {
                "fields": [{"name": "x" * 300, "type": "CharField"}],
                "relationships": [], "reverse_relationships": [],
            }
            for i in range(20)
        }
        search.return_value = {
            "matched_models": list(schemas),
            "expanded_models": [],
            "schemas": schemas,
        }

        result = search_database_schema.invoke({"query": "lead"})

        self.assertLessEqual(len(result), MAX_SCHEMA_TOOL_CHARS + 100)
        self.assertIn("omitted", result)

    @patch("ai_assistant.tools.search_schema_with_relationships")
    def test_schema_search_with_no_match_does_not_crash(self, search):
        search.return_value = {
            "matched_models": [], "expanded_models": [], "schemas": {}
        }
        self.assertEqual(
            search_database_schema.invoke({"query": "zzz"}),
            "No matching database model was found.",
        )

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

    def test_context_trimming_retains_latest_three_user_turns_for_follow_up(self):
        messages = [
            HumanMessage(content="An older unrelated request"),
            AIMessage(content="An older answer"),
            HumanMessage(content="An older company request"),
            AIMessage(content="An older company answer"),
            HumanMessage(content="Create Northstar Labs in California."),
            AIMessage(content="I have Northstar Labs in California. Which city?"),
            HumanMessage(content="San Francisco"),
        ]

        trimmed = trim_conversation_history(messages)

        self.assertEqual(MAX_HISTORY_TURNS, 3)
        self.assertEqual(trimmed, messages[2:])

    def test_agent_prompt_is_behavioural_and_lead_focused(self):
        messages = [HumanMessage(content="Create a lead for Ada, ada@example.com")]
        with patch("ai_assistant.agent.llm", autospec=True) as llm:
            llm.invoke.return_value = AIMessage(content="ok")
            agent_node({"messages": messages, "original_request": "make a lead for Ada"})

        prompt = llm.invoke.call_args.args[0]
        self.assertEqual(prompt[1:], messages)
        text = prompt[0].content
        self.assertIn("make a lead for Ada", text)       # original request kept
        self.assertIn("never bypass it", text.lower())   # approval rule
        for status in ("SUCCESS", "CANCELLED", "NEEDS_DETAILS", "ERROR"):
            self.assertIn(status, text)
        self.assertIn("cannot create or change any other kind of record", text)

    def test_prompt_does_not_hardcode_tool_signatures(self):
        """Tool names and parameters reach the model through bind_tools;
        repeating them in the prompt would drift out of date."""
        messages = [HumanMessage(content="hi")]
        with patch("ai_assistant.agent.llm", autospec=True) as llm:
            llm.invoke.return_value = AIMessage(content="ok")
            agent_node({"messages": messages, "original_request": ""})
        text = llm.invoke.call_args.args[0][0].content

        for tool_obj in tools:
            self.assertNotIn(tool_obj.name, text)
            for param in tool_obj.args:
                self.assertNotIn(f"{param},", text)
                self.assertNotIn(f"({param}", text)
        for removed in ("prepare_crm_bundle", "prepare_deal"):
            self.assertNotIn(removed, text)

    def test_tool_descriptions_carry_the_usage_details(self):
        """Since the prompt stays generic, the tool docstrings must explain
        what the model needs to know."""
        by_name = {tool_obj.name: tool_obj for tool_obj in tools}
        lead = by_name["prepare_lead"]
        self.assertIn("inherits", lead.description)
        self.assertIn("company_id", lead.description)
        self.assertIn("company_id", lead.args)
        for tool_obj in tools:
            self.assertTrue(tool_obj.description.strip(), tool_obj.name)

    def test_only_lead_write_tool_is_registered(self):
        names = {tool.name for tool in tools}
        self.assertEqual(
            names,
            {"search_database_schema", "list_departments",
             "search_companies", "search_contacts", "prepare_lead"},
        )
        self.assertEqual(
            {n for n in names if n.startswith("prepare_")}, {"prepare_lead"}
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

    def test_lead_rejection_creates_nothing(self):
        with (
            patch("ai_assistant.tools.interrupt", return_value="rejected"),
            patch("ai_assistant.tools._current_user",
                  return_value=SimpleNamespace(has_perm=lambda _p: True)),
        ):
            result = prepare_lead.invoke(
                {"first_name": "Rej", "email": "rej@example.com"},
                config={"configurable": {"user_id": 1}},
            )
        self.assertTrue(result.startswith("CANCELLED"))
        self.assertFalse(Lead.objects.filter(email="rej@example.com").exists())

    def test_lead_with_unknown_company_still_saves(self):
        user = get_user_model().objects.create_superuser(
            username="lead-admin", password="x", email="a@example.com"
        )
        with patch("ai_assistant.tools.interrupt", return_value="approved"):
            result = prepare_lead.invoke(
                {"first_name": "Grace Hopper", "email": "grace@nowhere.test",
                 "company_name": "No Such Company Ltd"},
                config={"configurable": {"user_id": user.pk}},
            )
        self.assertTrue(result.startswith("SUCCESS"), result)
        lead = Lead.objects.get(email="grace@nowhere.test")
        self.assertEqual((lead.first_name, lead.last_name), ("Grace", "Hopper"))
        self.assertIsNone(lead.company)

    def _admin(self, name="inherit-admin"):
        return get_user_model().objects.create_superuser(
            username=name, password="x", email=f"{name}@example.com"
        )

    def _run_lead(self, user, args, decision="approved"):
        """Run prepare_lead, returning (result, approval payload or None)."""
        seen = {}

        def fake_interrupt(payload):
            seen["payload"] = payload
            return decision

        with patch("ai_assistant.tools.interrupt", side_effect=fake_interrupt):
            result = prepare_lead.invoke(
                args, config={"configurable": {"user_id": user.pk}}
            )
        return result, seen.get("payload")

    def _company(self, **extra):
        country = Country.objects.create(name="USA", url_name="usa")
        city = City.objects.create(name="San Francisco", country=country)
        values = dict(
            full_name="Cyberdyne", country=country, city=city,
            city_name="San Francisco", phone="+1 415 555 0100",
            website="https://cyberdyne.example", address="1 Market St",
        )
        values.update(extra)
        return Company.objects.create(**values)

    def test_lead_inherits_company_fields_and_links_company(self):
        user = self._admin()
        company = self._company()

        result, payload = self._run_lead(user, {
            "first_name": "Rabia", "email": "rabia@gmail.com",
            "company_name": "cyberdyne", "phone": "0300 1234567",
        })

        self.assertTrue(result.startswith("SUCCESS"), result)
        lead = Lead.objects.get(email="rabia@gmail.com")
        self.assertEqual(lead.company, company)
        self.assertEqual(lead.company_name, "Cyberdyne")
        self.assertEqual(lead.country, company.country)
        self.assertEqual(lead.city, company.city)
        self.assertEqual(lead.city_name, "San Francisco")
        self.assertEqual(lead.company_phone, "+1 415 555 0100")
        self.assertEqual(lead.website, "https://cyberdyne.example")
        self.assertEqual(lead.company_address, "1 Market St")
        # the person's own phone is never replaced by the company's
        self.assertEqual(lead.phone, "0300 1234567")
        self.assertEqual(lead.owner, user)
        # approval card tells the user what was filled in automatically
        self.assertEqual(payload["inferred"]["Country"], "USA")
        self.assertEqual(payload["inferred"]["City"], "San Francisco")
        self.assertEqual(payload["inferred"]["Owner"], "inherit-admin")
        self.assertIn("Cyberdyne", payload["related"]["Company match"])
        self.assertEqual(payload["warnings"], [])

    def test_lead_without_user_phone_leaves_person_phone_blank(self):
        user = self._admin("inherit-admin-2")
        self._company()
        self._run_lead(user, {
            "first_name": "Sam", "email": "sam@example.com",
            "company_name": "Cyberdyne",
        })
        lead = Lead.objects.get(email="sam@example.com")
        self.assertEqual(lead.phone, "")
        self.assertEqual(lead.company_phone, "+1 415 555 0100")

    def test_unknown_company_warns_and_inherits_nothing(self):
        user = self._admin("inherit-admin-3")
        result, payload = self._run_lead(user, {
            "first_name": "Ann", "email": "ann@example.com",
            "company_name": "Skynet",
        })
        self.assertTrue(result.startswith("SUCCESS"), result)
        self.assertTrue(any("Skynet" in w for w in payload["warnings"]))
        self.assertEqual(payload["related"], {})
        self.assertEqual(list(payload["inferred"]), ["Owner"])
        lead = Lead.objects.get(email="ann@example.com")
        self.assertIsNone(lead.company)
        self.assertEqual(lead.company_name, "Skynet")
        self.assertIsNone(lead.country)

    def test_ambiguous_company_asks_for_details_without_approval(self):
        user = self._admin("inherit-admin-4")
        first = self._company()
        other_country = Country.objects.create(name="Canada", url_name="canada")
        second = Company.objects.create(full_name="Cyberdyne", country=other_country)

        with patch("ai_assistant.tools.interrupt") as interrupt:
            result = prepare_lead.invoke(
                {"first_name": "Al", "email": "al@example.com",
                 "company_name": "Cyberdyne"},
                config={"configurable": {"user_id": user.pk}},
            )
        interrupt.assert_not_called()
        self.assertTrue(result.startswith("NEEDS_DETAILS"), result)
        self.assertIn(f"#{first.pk}", result)
        self.assertIn(f"#{second.pk}", result)

        result, _ = self._run_lead(user, {
            "first_name": "Al", "email": "al@example.com",
            "company_name": "Cyberdyne", "company_id": second.pk,
        })
        self.assertTrue(result.startswith("SUCCESS"), result)
        self.assertEqual(Lead.objects.get(email="al@example.com").company, second)

    def test_unknown_company_id_asks_for_details(self):
        user = self._admin("inherit-admin-5")
        with patch("ai_assistant.tools.interrupt") as interrupt:
            result = prepare_lead.invoke(
                {"first_name": "Bo", "email": "bo@example.com", "company_id": 999999},
                config={"configurable": {"user_id": user.pk}},
            )
        interrupt.assert_not_called()
        self.assertTrue(result.startswith("NEEDS_DETAILS"))

    def test_company_values_too_long_for_lead_are_skipped_with_warning(self):
        user = self._admin("inherit-admin-6")
        self._company(phone="1" * 50, city_name="x" * 80)
        result, payload = self._run_lead(user, {
            "first_name": "Cy", "email": "cy@example.com",
            "company_name": "Cyberdyne",
        })
        self.assertTrue(result.startswith("SUCCESS"), result)
        lead = Lead.objects.get(email="cy@example.com")
        self.assertEqual(lead.company_phone, "")
        self.assertEqual(lead.city_name, "")
        self.assertEqual(len(payload["warnings"]), 2)
        self.assertEqual(lead.country.name, "USA")

    def test_duplicate_email_is_rejected_before_the_approval_card(self):
        user = self._admin("inherit-admin-7")
        Lead.objects.create(first_name="Old", email="old@example.com", owner=user)
        with patch("ai_assistant.tools.interrupt") as interrupt:
            result = prepare_lead.invoke(
                {"first_name": "New", "email": "OLD@example.com"},
                config={"configurable": {"user_id": user.pk}},
            )
        interrupt.assert_not_called()
        self.assertTrue(result.startswith("CANCELLED"))

    def test_rejected_lead_with_company_creates_nothing(self):
        user = self._admin("inherit-admin-8")
        self._company()
        result, _ = self._run_lead(user, {
            "first_name": "No", "email": "no2@example.com",
            "company_name": "Cyberdyne",
        }, decision="rejected")
        self.assertTrue(result.startswith("CANCELLED"))
        self.assertFalse(Lead.objects.filter(email="no2@example.com").exists())

    def test_duplicate_lead_email_is_cancelled(self):
        user = get_user_model().objects.create_superuser(
            username="lead-admin2", password="x", email="b@example.com"
        )
        Lead.objects.create(first_name="Dup", email="dup@example.com", owner=user)
        with patch("ai_assistant.tools.interrupt", return_value="approved"):
            result = prepare_lead.invoke(
                {"first_name": "Dup2", "email": "DUP@example.com"},
                config={"configurable": {"user_id": user.pk}},
            )
        self.assertTrue(result.startswith("CANCELLED"), result)

    def test_lead_without_permission_raises_before_approval(self):
        from django.core.exceptions import PermissionDenied
        with (
            patch("ai_assistant.tools.interrupt") as interrupt,
            patch("ai_assistant.tools._current_user",
                  return_value=SimpleNamespace(has_perm=lambda _p: False)),
        ):
            with self.assertRaises(PermissionDenied):
                prepare_lead.invoke(
                    {"first_name": "No", "email": "no@example.com"},
                    config={"configurable": {"user_id": 1}},
                )
        interrupt.assert_not_called()

    def test_company_and_contact_lookups_handle_empty_and_unknown(self):
        user = get_user_model().objects.create_user(username="reader", password="x")
        cfg = {"configurable": {"user_id": user.pk}}
        self.assertIn("No companies", search_companies.invoke({"name": "zzz"}, config=cfg))
        self.assertTrue(
            search_contacts.invoke({"company_name": "zzz"}, config=cfg)
            .startswith("NEEDS_DETAILS")
        )
        self.assertIn("No contacts", search_contacts.invoke({"name": "zzz"}, config=cfg))

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
        return_value=SimpleNamespace(
            values={"messages": ["started"]},
            tasks=[SimpleNamespace(interrupts=[SimpleNamespace(
                value={
                    "action": "CREATE_LEAD",
                    "data": {"first_name": "Ada", "email": "ada@example.com"},
                }
            )])],
        ),
    )
    @patch("ai_assistant.api_views.app.invoke", return_value={})
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
        self.assertEqual(get_state.call_count, 2)

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


class TestLlmFallback(TestCase):
    def test_gemini_bindings_do_not_forward_parallel_tool_calls(self):
        for model in (gemini_primary, gemini_fallback):
            self.assertNotIn("parallel_tool_calls", model.kwargs)

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
