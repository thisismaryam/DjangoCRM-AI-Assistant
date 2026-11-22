import json
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.core.exceptions import ValidationError
from django.db.models import Q
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ai_assistant.rag.search import search_schema_with_relationships
from common.models import Department
from crm.models import Company, Contact, Lead

SCHEMA_CATALOG_PATH = Path(__file__).resolve().parent / \
    "data" / "schema_catalog.json"

# Phase 2.7 — bounded-output constants for the schema tool
MAX_SCHEMA_FIELDS_PER_MODEL = 45
MAX_SCHEMA_RELATIONSHIPS_PER_MODEL = 10
MAX_SCHEMA_TOOL_CHARS = 4500

# System/audit fields the assistant never fills in; hidden from schema answers.
HIDDEN_SCHEMA_FIELDS = {
    "creation_date", "update_date", "modified_by", "token", "avatar",
}

with SCHEMA_CATALOG_PATH.open(encoding="utf-8") as schema_file:
    SCHEMA_CATALOG = json.load(schema_file)


# ------------------------------------------------------------------
# Phase 3 — standard approval + return contract
# ------------------------------------------------------------------

def _approval_payload(
    action: str,
    record_type: str,
    data: dict,
    related: dict | None = None,
    warnings: list[str] | None = None,
    inferred: dict | None = None,
) -> dict:
    """Canonical shape passed to `interrupt(...)`. The frontend renders one
    approval card regardless of which model is being created."""
    return {
        "action": action,
        "record_type": record_type,
        "data": data,
        "related": related or {},
        "warnings": warnings or [],
        # Values the tool filled in itself (not typed by the user).
        "inferred": inferred or {},
    }


def _ok(message: str) -> str:
    return f"SUCCESS: {message}"


def _cancelled(message: str) -> str:
    return f"CANCELLED: {message}"


def _needs_details(message: str) -> str:
    return f"NEEDS_DETAILS: {message}"


def _error(message: str) -> str:
    return f"ERROR: {message}"


def _current_user(config: RunnableConfig) -> Any:
    user_id = config.get("configurable", {}).get("user_id")
    if user_id is None:
        raise PermissionDenied(
            "An authenticated user is required for this tool.")
    user = get_user_model().objects.filter(pk=user_id, is_active=True).first()
    if user is None:
        raise PermissionDenied("The authenticated user is no longer active.")
    return user


def _require_permission(user: Any, permission: str) -> None:
    if not user.has_perm(permission):
        raise PermissionDenied(f"Permission required: {permission}.")


def _named_record(model: Any, name: str, **filters: Any) -> Any | None:
    candidates = model.objects.filter(**filters).filter(
        Q(name__iexact=name) | Q(alternative_names__icontains=name)
    )
    matching = [
        item
        for item in candidates
        if item.name.casefold() == name.casefold()
        or name.casefold()
        in {
            alias.strip().casefold()
            for alias in (item.alternative_names or "").split(",")
        }
    ]
    if len(matching) > 1:
        raise ValidationError(
            f'More than one {model._meta.verbose_name} matches "{name}".'
        )
    return matching[0] if matching else None



@tool
def list_departments(*, config: RunnableConfig) -> str:
    """List the departments associated with the authenticated user."""
    user = _current_user(config)
    departments = Department.objects.filter(
        user=user).distinct().order_by("name")
    return json.dumps(
        [
            {"id": department.pk, "name": department.name}
            for department in departments
        ],
        ensure_ascii=False,
    )

def _format_field(field: dict, relationships: dict) -> str:
    """One compact field description, e.g. `email*`, `country -> Country`,
    `company (Company of contact) -> Company`, `tags -> [Tag]`."""
    text = field["name"]
    if field.get("verbose_name"):
        text += f" ({field['verbose_name']})"
    if field["name"] in relationships:
        rel = relationships[field["name"]]
        target = rel["related_model"]
        text += f" -> [{target}]" if rel["type"] == "ManyToMany" else f" -> {target}"
    else:
        text += f": {field['type']}"
    if field.get("required_for_input"):
        text += "*"
    return text


def _format_schema_entry(name: str, entry: dict, is_matched: bool) -> str:
    """Render one model entry compactly.

    Matched models list their business fields (required first, audit fields
    hidden, foreign keys shown with their target model). Expanded models
    only list relationships, to save context."""
    verbose = entry.get("verbose_name", name)
    if verbose and verbose.lower() != name.lower():
        lines = [f"## {name} ({verbose})"]
    else:
        lines = [f"## {name}"]

    forward = entry.get("relationships", [])
    reverse = entry.get("reverse_relationships", [])[
        :MAX_SCHEMA_RELATIONSHIPS_PER_MODEL
    ]

    if is_matched:
        relationships = {r["field"]: r for r in forward}
        visible = [
            f for f in entry.get("fields", [])
            if f["name"] not in HIDDEN_SCHEMA_FIELDS
        ]
        # Required fields first; sorted() is stable so catalog order is kept.
        visible = sorted(
            visible, key=lambda f: not f.get("required_for_input"))
        shown = visible[:MAX_SCHEMA_FIELDS_PER_MODEL]
        if shown:
            field_lines = ", ".join(
                _format_field(f, relationships) for f in shown
            )
            remaining = len(visible) - len(shown)
            if remaining > 0:
                field_lines += f" [+{remaining} more]"
            lines.append(f"Fields (* = required): {field_lines}")
    else:
        forward = [
            r for r in forward if r["field"] not in HIDDEN_SCHEMA_FIELDS
        ][:MAX_SCHEMA_RELATIONSHIPS_PER_MODEL]
        if forward:
            rel_lines = "; ".join(
                f"{r['field']} -> {r['related_model']} ({r['type']})"
                for r in forward
            )
            lines.append(f"Relationships: {rel_lines}")

    if reverse:
        rev_lines = "; ".join(
            f"{r['related_model']} via {r['field']}"
            for r in reverse
        )
        lines.append(f"Referenced by: {rev_lines}")

    return "\n".join(lines)


@tool
def search_database_schema(query: str) -> str:
    """Look up CRM model schemas relevant to the query.

    Returns matched models with their fields and relationships, plus directly
    connected models showing how they are linked. Use this before planning any
    query or write operation against unfamiliar data, and before claiming a
    field or relationship exists.
    """
    result = search_schema_with_relationships(query, top_k=2)
    if not result["schemas"]:
        return "No matching database model was found."

    matched = set(result["matched_models"])
    ordered = list(result["matched_models"]) + [
        n for n in result["expanded_models"] if n not in matched
    ]

    blocks = []
    total_chars = 0

    for name in ordered:
        entry = result["schemas"].get(name)
        if not entry:
            continue

        block = _format_schema_entry(
            name, entry, is_matched=(name in matched))

        if total_chars + len(block) + 2 > MAX_SCHEMA_TOOL_CHARS:
            blocks.append(
                "[additional related models omitted to fit context limit]")
            break

        blocks.append(block)
        total_chars += len(block) + 2

    return "\n\n".join(blocks)


class _CompanyLookupError(Exception):
    """The company could not be resolved to exactly one record."""


def _resolve_company(company_name: str, company_id: int) -> Company | None:
    """Find the CRM Company a lead refers to.

    Returns None when there is no match (a new, unknown company is fine).
    Raises _CompanyLookupError when the reference is ambiguous or invalid.
    """
    if company_id:
        company = Company.objects.select_related(
            "country", "city").filter(pk=company_id).first()
        if company is None:
            raise _CompanyLookupError(
                f"No company with id {company_id} exists. Use "
                "search_companies to find the right id."
            )
        return company

    name = (company_name or "").strip()
    if not name:
        return None

    candidates = Company.objects.select_related("country", "city").filter(
        Q(full_name__iexact=name) | Q(alternative_names__icontains=name)
    )
    matches = [
        c for c in candidates
        if c.full_name.casefold() == name.casefold()
        or name.casefold() in {
            alias.strip().casefold()
            for alias in (c.alternative_names or "").split(",")
        }
    ]
    if len(matches) > 1:
        options = "; ".join(
            f"#{c.pk} {c.full_name}"
            f" ({c.country.name if c.country_id else 'no country'})"
            for c in matches[:5]
        )
        raise _CompanyLookupError(
            f'More than one company matches "{name}": {options}. '
            "Ask the user which one, then call prepare_lead again with "
            "that company_id."
        )
    return matches[0] if matches else None


def _fit(value: str, max_length: int, label: str, warnings: list[str]) -> str:
    """Return value if it fits the Lead field, else '' plus a warning."""
    value = (value or "").strip()
    if len(value) > max_length:
        warnings.append(
            f"The company's {label} is too long for a lead "
            f"(max {max_length} characters), so it was not copied."
        )
        return ""
    return value


def _company_defaults(company: Company, warnings: list[str]) -> dict:
    """Lead field values inherited from an existing Company."""
    return {
        "country": company.country,
        "city": company.city,
        "city_name": _fit(company.city_name, 50, "city name", warnings),
        "company_phone": _fit(company.phone, 20, "phone number", warnings),
        "website": _fit(company.website, 200, "website", warnings),
        "company_address": (company.address or "").strip(),
    }


def _user_label(user: Any) -> str:
    full_name = getattr(user, "get_full_name", lambda: "")()
    return full_name or getattr(user, "username", "") or str(user)


@tool
def prepare_lead(
    first_name: str,
    email: str,
    company_name: str = "",
    last_name: str = "",
    phone: str = "",
    company_id: int = 0,
    *,
    config: RunnableConfig,
) -> str:
    """Prepare a new Lead for human approval.

    Only use this when the user explicitly wants to add a new lead.
    `phone` is the person's own phone. If `company_name` matches exactly one
    existing CRM company, the lead is linked to it and inherits its country,
    city, phone, website and address automatically. If several companies share
    the name, pass the chosen `company_id` (from search_companies).
    """

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

    user = _current_user(config)
    _require_permission(user, "crm.add_lead")

    # Reject a duplicate before bothering the user with an approval card.
    if Lead.objects.filter(email__iexact=lead_data["email"]).exists():
        return _cancelled(
            f"A lead with the email {lead_data['email']} already exists in the CRM database."
        )

    try:
        company = _resolve_company(company_name, company_id)
    except _CompanyLookupError as exc:
        return _needs_details(str(exc))

    warnings: list[str] = []
    inherited: dict = {}
    related: dict = {}
    inferred: dict = {"Owner": _user_label(user)}
    if company is not None:
        lead_data["company_name"] = company.full_name
        inherited = _company_defaults(company, warnings)
        related["Company match"] = f"#{company.pk} {company.full_name}"
        for label, value in (
            ("Country", inherited["country"]),
            ("City", inherited["city"] or inherited["city_name"]),
            ("Company phone", inherited["company_phone"]),
            ("Website", inherited["website"]),
            ("Company address", inherited["company_address"]),
        ):
            if value:
                inferred[label] = str(value)
    elif lead_data["company_name"]:
        warnings.append(
            f'"{lead_data["company_name"]}" is not in the CRM; the lead will '
            "store the company name only."
        )

    # 1. Pause the graph and hand the data to the frontend
    user_decision = interrupt(
        _approval_payload(
            action="CREATE_LEAD",
            record_type="Lead",
            data={k: v for k, v in lead_data.items() if v},
            related=related,
            warnings=warnings,
            inferred=inferred,
        )
    )

    # 2. This line ONLY runs after the graph is resumed with Command(resume=...)
    if user_decision == "approved":
        user = _current_user(config)
        _require_permission(user, "crm.add_lead")
        try:
            # Check again: another lead may have been added while waiting.
            if Lead.objects.filter(email__iexact=lead_data["email"]).exists():
                return _cancelled(
                    f"A lead with the email {lead_data['email']} already exists in the CRM database."
                )

            new_lead = Lead.objects.create(
                first_name=lead_data["first_name"],
                last_name=lead_data["last_name"],
                email=lead_data["email"],
                company_name=lead_data["company_name"],
                phone=lead_data["phone"],
                company=company,
                owner=user,
                **inherited,
            )
            return _ok(
                f"Lead for {new_lead.first_name} created. "
                f"<a href='/en/456-admin/crm/lead/{new_lead.id}/change/' "
                f"target='_blank'>Click here to view the Lead in CRM</a>."
            )

        except Exception as e:
            return _error(
                f"Could not save lead to database. Details: {str(e)}"
            )
    else:
        return _cancelled("The user rejected the creation of this lead.")


@tool
def search_companies(name: str = "", *, config: RunnableConfig) -> str:
    """Search CRM companies by partial name (case-insensitive).

    Returns up to 20 companies with id, full_name, country, city, and
    phone. Use this before any operation that requires a company ID.
    """
    _current_user(config)

    qs = Company.objects.select_related(
        "country", "city").order_by("full_name")
    if name.strip():
        qs = qs.filter(full_name__icontains=name.strip())

    rows = list(qs[:20])
    if not rows:
        return f'No companies found matching "{name}".' if name else \
            "No companies in the database."

    payload = [
        {
            "id": c.pk,
            "name": c.full_name,
            "country": c.country.name if c.country_id else None,
            "city": c.city_name or (c.city.name if c.city_id else None),
            "phone": c.phone or None,
        }
        for c in rows
    ]
    return json.dumps({"count": len(payload), "companies": payload},
                      ensure_ascii=False)




MAX_CONTACT_RESULTS = 25


def _contact_to_dict(contact: Contact, *, include_related: bool = False) -> dict:
    """Serialise a Contact without leaking model objects."""
    payload = {
        "id": contact.pk,
        "full_name": contact.full_name,
        "email": contact.email,
        "phone": contact.phone or None,
        "title": contact.title or None,
        "company": (
            {"id": contact.company_id, "name": contact.company.full_name}
            if contact.company_id else None
        ),
    }
    if include_related:
        payload["description"] = contact.description or None
        payload["address"] = contact.address or None
        payload["city_name"] = contact.city_name or None
        payload["disqualified"] = contact.disqualified
    return payload


@tool
def search_contacts(
    company_name: str = "",
    name: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Search CRM contacts by company name or contact name.

    All parameters are optional. 
    Returns up to 25 contacts with id, full_name, email, and company.
    Use get_contact_details for full info.
    """
    _current_user(config)

    qs = Contact.objects.select_related(
        "company").order_by("first_name", "last_name")

    if company_name.strip():
        company = Company.objects.filter(
            full_name__iexact=company_name.strip()
        ).first()
        if company is None:
            return _needs_details(f'No company found with name "{company_name}".')
        qs = qs.filter(company=company)

    if name.strip():
        # Search first, middle, or last name
        qs = qs.filter(
            Q(first_name__icontains=name.strip()) |
            Q(last_name__icontains=name.strip())
        )

    results = list(qs[:MAX_CONTACT_RESULTS])
    if not results:
        return "No contacts match the given criteria."

    payload = [_contact_to_dict(c) for c in results]
    return json.dumps(
        {"count": len(payload), "contacts": payload},
        ensure_ascii=False,
    )




tools = [
    search_database_schema,
    list_departments,
    search_companies,
    search_contacts,
    prepare_lead,
]
