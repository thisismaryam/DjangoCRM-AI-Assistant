from unittest.mock import patch

from django.test import TestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_assistant.context import MAX_HISTORY_CHARS, trim_conversation_history
from ai_assistant.tools import (
    MAX_SCHEMA_RESULT_CHARS,
    prepare_lead,
    search_database_schema,
)


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
            patch("ai_assistant.tools.get_user_model") as get_user_model,
            patch("ai_assistant.tools.Lead.objects.filter") as filter_leads,
            patch("ai_assistant.tools.Lead.objects.create", return_value=lead),
        ):
            filter_leads.return_value.exists.return_value = False
            user_model = get_user_model.return_value
            user_model.objects.filter.return_value.first.return_value = None

            result = prepare_lead.invoke(
                {"first_name": "Ada", "email": "ada@example.com"}
            )

        self.assertIn("SUCCESS: Lead for Ada created.", result)
        self.assertIn(
            "<a href='/en/123/crm/lead/42/change/' target='_blank'>"
            "Click here to view the Lead in CRM</a>",
            result,
        )
