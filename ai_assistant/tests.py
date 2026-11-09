import json
from unittest.mock import patch

from django.test import TestCase

from ai_assistant.tools import (
    SCHEMA_CATALOG,
    prepare_lead,
    search_database_schema,
)


class TestAssistantTools(TestCase):
    @patch(
        "ai_assistant.tools.search_schema",
        return_value=[{"model": "Lead"}, {"model": "Deal"}],
    )
    def test_schema_search_returns_only_top_model_catalog_entry(self, search_schema):
        result = search_database_schema.invoke({"query": "lead contact"})

        self.assertEqual(json.loads(result), {"Lead": SCHEMA_CATALOG["Lead"]})
        search_schema.assert_called_once_with("lead contact", top_k=1)

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
