from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.contrib.auth.models import Permission
from django.core.exceptions import PermissionDenied
from django.db import IntegrityError
from django.utils import timezone
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from rest_framework.test import APIRequestFactory, force_authenticate

from ai_assistant.api_views import approve_view, chat_view
from ai_assistant.context import MAX_HISTORY_CHARS, trim_conversation_history
from ai_assistant.llm import build_fallback_llm
from ai_assistant.task_tools import (
    list_projects,
    list_team_members,
    list_tasks,
    prepare_project,
    prepare_subtask,
    prepare_task,
)
from ai_assistant.tools import (
    MAX_SCHEMA_RESULT_CHARS,
    prepare_lead,
    search_database_schema,
    tools,
)
from common.models import Reminder
from tasks.models import Project, Task, TaskStage


class TestAssistantTools(TestCase):
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


class TestTaskManagementTools(TestCase):
    def setUp(self):
        Group.objects.get_or_create(name="co-workers")
        self.user = get_user_model().objects.create_user(
            username="task-user",
            password="test-password",
        )
        permissions = Permission.objects.filter(
            content_type__app_label="tasks",
            codename__in=(
                "view_project",
                "view_task",
                "add_project",
                "add_task",
            ),
        )
        self.user.user_permissions.add(*permissions)
        self.config = {"configurable": {"user_id": self.user.pk}}
        self.stage = TaskStage.objects.create(name="To do", default=True)
        self.project = Project.objects.create(
            name="CRM upgrade",
            next_step="Review project progress",
            next_step_date=timezone.localdate(),
            owner=self.user,
        )

    def test_read_only_lookup_tools_do_not_interrupt(self):
        with patch("ai_assistant.task_tools.interrupt") as interrupt:
            list_projects.invoke({"query": "CRM"}, config=self.config)
            list_team_members.invoke({"query": ""}, config=self.config)
            list_tasks.invoke(
                {"project_id": self.project.pk}, config=self.config
            )

        interrupt.assert_not_called()

    @patch("ai_assistant.task_tools.interrupt", return_value="rejected")
    def test_prepare_project_rejection_does_not_create_project(self, interrupt):
        before = Project.objects.count()

        result = prepare_project.invoke(
            {"name": "Rejected project"},
            config=self.config,
        )

        self.assertIn("CANCELLED", result)
        self.assertEqual(Project.objects.count(), before)
        self.assertEqual(interrupt.call_count, 1)

    @patch("ai_assistant.task_tools.interrupt", return_value="approved")
    def test_prepare_project_applies_assignees_tags_and_reminder(self, interrupt):
        result = prepare_project.invoke(
            {
                "name": "New CRM project",
                "assignee_ids": [self.user.pk],
                "tags": ["launch"],
                "reminder_date": "2026-12-01",
                "reminder_description": "Project checkpoint",
            },
            config=self.config,
        )

        project = Project.objects.get(name="New CRM project")
        self.assertIn(str(project.pk), result)
        self.assertEqual(list(project.responsible.all()), [self.user])
        self.assertEqual(list(project.tags.values_list("name", flat=True)), ["launch"])
        reminder = Reminder.objects.get(object_id=project.pk)
        self.assertEqual(reminder.reminder_date.date().isoformat(), "2026-12-01")
        self.assertEqual(interrupt.call_count, 1)

    @patch("ai_assistant.task_tools.interrupt", return_value="approved")
    def test_prepare_task_and_subtask_enforce_project_hierarchy(self, interrupt):
        task_result = prepare_task.invoke(
            {"project_id": self.project.pk, "name": "Plan rollout"},
            config=self.config,
        )
        task = Task.objects.get(name="Plan rollout")
        self.assertIn(str(task.pk), task_result)
        self.assertEqual(task.project_id, self.project.pk)
        self.assertIsNone(task.task_id)

        subtask_result = prepare_subtask.invoke(
            {"parent_task_id": task.pk, "name": "Review migration"},
            config=self.config,
        )
        subtask = Task.objects.get(name="Review migration")
        self.assertIn(str(subtask.pk), subtask_result)
        self.assertEqual(subtask.project_id, self.project.pk)
        self.assertEqual(subtask.task_id, task.pk)
        self.assertEqual(interrupt.call_count, 2)

        nested_result = prepare_subtask.invoke(
            {"parent_task_id": subtask.pk, "name": "Unsupported nested item"},
            config=self.config,
        )
        self.assertIn("cannot have nested subtasks", nested_result)
        self.assertEqual(interrupt.call_count, 2)

    def test_prepare_task_rejects_unknown_project_and_non_iso_dates(self):
        unknown_project = prepare_task.invoke(
            {"project_id": 999999, "name": "Task"},
            config=self.config,
        )
        invalid_date = prepare_project.invoke(
            {"name": "Invalid date project", "due_date": "next Friday"},
            config=self.config,
        )

        self.assertIn("list_projects", unknown_project)
        self.assertIn("YYYY-MM-DD", invalid_date)
        self.assertFalse(Project.objects.filter(name="Invalid date project").exists())

    def test_prepare_task_rechecks_project_visibility_after_approval(self):
        other_user = get_user_model().objects.create_user(
            username="other-owner",
        )

        def revoke_project_access(_payload):
            Project.objects.filter(pk=self.project.pk).update(owner=other_user)
            return "approved"

        before = Task.objects.count()
        with patch(
            "ai_assistant.task_tools.interrupt",
            side_effect=revoke_project_access,
        ):
            with self.assertRaises(PermissionDenied):
                prepare_task.invoke(
                    {"project_id": self.project.pk, "name": "No longer accessible"},
                    config=self.config,
                )

        self.assertEqual(Task.objects.count(), before)

    def test_prepare_task_rechecks_add_permission_after_approval(self):
        add_task = Permission.objects.get(
            content_type__app_label="tasks",
            codename="add_task",
        )

        def revoke_add_permission(_payload):
            self.user.user_permissions.remove(add_task)
            return "approved"

        before = Task.objects.count()
        with patch(
            "ai_assistant.task_tools.interrupt",
            side_effect=revoke_add_permission,
        ):
            with self.assertRaises(PermissionDenied):
                prepare_task.invoke(
                    {"project_id": self.project.pk, "name": "No longer permitted"},
                    config=self.config,
                )

        self.assertEqual(Task.objects.count(), before)

    def test_write_requires_task_add_permission(self):
        user = get_user_model().objects.create_user(
            username="read-only",
            password="test-password",
        )
        user.user_permissions.add(
            *Permission.objects.filter(
                content_type__app_label="tasks",
                codename="view_project",
            )
        )

        with patch("ai_assistant.task_tools.interrupt") as interrupt:
            with self.assertRaises(PermissionDenied):
                prepare_task.invoke(
                    {"project_id": self.project.pk, "name": "Unauthorized"},
                    config={"configurable": {"user_id": user.pk}},
                )

        interrupt.assert_not_called()

    @patch(
        "ai_assistant.api_views.app.get_state",
        return_value=SimpleNamespace(values={"messages": ["started"]}),
    )
    @patch(
        "ai_assistant.api_views.app.invoke",
        return_value={
            "__interrupt__": [
                SimpleNamespace(value={"action": "CREATE_TASK", "data": {"name": "Task"}})
            ]
        },
    )
    def test_approval_endpoint_returns_next_pending_write(
        self, invoke, get_state
    ):
        request = APIRequestFactory().post(
            "/api/ai/approve/",
            {"thread_id": "project-workflow", "decision": "approved"},
            format="json",
        )
        force_authenticate(
            request,
            user=SimpleNamespace(pk=73, is_authenticated=True, is_active=True),
        )

        response = approve_view(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "pending_approval")
        self.assertEqual(response.data["thread_id"], "project-workflow")
        self.assertEqual(response.data["payload"]["action"], "CREATE_TASK")
        invoke.assert_called_once()
        self.assertEqual(
            invoke.call_args.kwargs["config"]["configurable"],
            {"thread_id": "user-73:project-workflow", "user_id": 73},
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
        names = {tool.name for tool in tools}
        self.assertTrue(
            {
                "prepare_lead",
                "list_projects",
                "list_team_members",
                "list_tasks",
                "prepare_project",
                "prepare_task",
                "prepare_subtask",
            }.issubset(names)
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

        response = build_fallback_llm(primary, fallback).invoke(
            "Create a lead for Ada."
        )

        self.assertEqual(response.tool_calls[0]["name"], "prepare_lead")
