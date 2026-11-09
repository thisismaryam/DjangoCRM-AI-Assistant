import json
from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import interrupt
from ai_assistant.rag.search import search_schema
from crm.models import Lead
from django.contrib.auth import get_user_model


SCHEMA_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "schema_catalog.json"
with SCHEMA_CATALOG_PATH.open(encoding="utf-8") as schema_file:
    SCHEMA_CATALOG = json.load(schema_file)


@tool
def search_database_schema(query: str) -> str:
    """Useful for when you need to know which database model or table stores specific information."""
    results = search_schema(query, top_k=1)
    if not results:
        return "No matching database model was found."

    model_name = results[0]["model"]
    return json.dumps(
        {model_name: SCHEMA_CATALOG[model_name]},
        separators=(",", ":"),
        ensure_ascii=False,
    )


@tool
def prepare_lead(first_name: str, email: str, company_name: str = "", last_name: str = "", phone: str = "") -> str:
    """Useful for when the user asks to create a new lead. 
    Only use this when the user explicitly wants to add a new lead to the system."""

    # Split first/last name if user gave full name in first_name
    first = first_name.strip()
    last = last_name.strip()
    if not last and " " in first:
        parts = first.split(" ", 1)
        first, last = parts[0], parts[1]

    lead_data = {
        "first_name": first,
        "last_name": last,
        "email": email.strip(),
        "company_name": company_name.strip(),
        "phone": phone.strip()
    }

    # 1. Pause the graph and hand the data to the frontend
    user_decision = interrupt({
        "action": "CREATE_LEAD",
        "data": {k: v for k, v in lead_data.items() if v}
    })

    # 2. This line ONLY runs after the graph is resumed with Command(resume=...)
    if user_decision == "approved":
        try:
            # Check for duplicates first (case-insensitive)
            if Lead.objects.filter(email__iexact=lead_data["email"]).exists():
                return f"CANCELLED: A lead with the email {lead_data['email']} already exists in the CRM database."

            User = get_user_model()
            default_owner = User.objects.filter(is_superuser=True).first()

            # Actually save to the database using Django ORM
            new_lead = Lead.objects.create(
                first_name=lead_data["first_name"],
                last_name=lead_data["last_name"],
                email=lead_data["email"],
                company_name=lead_data["company_name"],
                phone=lead_data["phone"],
                owner=default_owner
            )
            return (
                f"SUCCESS: Lead for {new_lead.first_name} created. "
                f"<a href='/en/123/crm/lead/{new_lead.id}/change/' "
                f"target='_blank'>Click here to view the Lead in CRM</a>."
            )

        except Exception as e:
            return f"ERROR: Could not save lead to database. Details: {str(e)}"
    else:
        return "CANCELLED: The user rejected the creation of this lead."


tools = [search_database_schema, prepare_lead]
