import json
from pathlib import Path
from typing import Any
from django.db import models
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.utils.text import slugify
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from ai_assistant.rag.search import search_schema, search_schema_with_relationships
from common.models import Department
from crm.models import City, ClientType, Company, Country, Lead
from crm.models import (
    City, ClientType, Company, Country, Lead,
    Currency, Deal, Payment, Rate,
)

SCHEMA_CATALOG_PATH = Path(__file__).resolve().parent / \
    "data" / "schema_catalog.json"
MAX_SCHEMA_FIELDS = 10
MAX_SCHEMA_RESULT_CHARS = 1000

# Phase 2.7 — bounded-output constants for the schema tool
MAX_SCHEMA_FIELDS_PER_MODEL = 8
MAX_SCHEMA_RELATIONSHIPS_PER_MODEL = 10
MAX_SCHEMA_TOOL_CHARS = 2200

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
) -> dict:
    """Canonical shape passed to `interrupt(...)`. The frontend renders one
    approval card regardless of which model is being created."""
    return {
        "action": action,
        "record_type": record_type,
        "data": data,
        "related": related or {},
        "warnings": warnings or [],
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


def _user_departments(user: Any) -> list[Any]:
    return list(Department.objects.filter(user=user).distinct())


class _CompanyDepartmentError(Exception):
    pass


def _matching_client_type(user: Any, name: str) -> Any | None:
    matches = list(ClientType.objects.filter(name__iexact=name))
    if len(matches) <= 1:
        return matches[0] if matches else None

    departments = _user_departments(user)
    if len(departments) == 1:
        department_matches = [
            item for item in matches if item.department_id == departments[0].pk
        ]
        if len(department_matches) == 1:
            return department_matches[0]
    raise ValidationError(
        f'More than one company type matches "{name}". Please clarify the type.'
    )


def _company_reference_state(
    user: Any,
    country_name: str,
    city_name: str,
    type_name: str,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    country = _named_record(Country, country_name) if country_name else None
    if city_name and not country_name:
        raise ValidationError("A country is required when a city is provided.")
    city = (
        _named_record(City, city_name, country=country)
        if city_name and country
        else None
    )
    client_type = _matching_client_type(user, type_name) if type_name else None
    department = None
    if type_name and client_type is None:
        departments = Department.objects.filter(user=user).distinct()
        department_count = departments.count()
        if department_count != 1:
            raise _CompanyDepartmentError(
                f"Cannot create company type '{type_name}': your account has "
                f"{department_count} departments; exactly one is required."
            )
        department = Department.objects.filter(user=user).first()
    return country, city, client_type, department


def _require_company_permissions(
    user: Any,
    country_name: str,
    city_name: str,
    type_name: str,
    references: tuple[Any | None, Any | None, Any | None, Any | None],
) -> None:
    _require_permission(user, "crm.add_company")
    if country_name and references[0] is None:
        _require_permission(user, "crm.add_country")
    if city_name and references[1] is None:
        _require_permission(user, "crm.add_city")
    if type_name and references[2] is None:
        _require_permission(user, "crm.add_clienttype")


def _get_or_create_company_references(
    user: Any,
    country_name: str,
    city_name: str,
    type_name: str,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    country, city, client_type, department = _company_reference_state(
        user, country_name, city_name, type_name
    )
    _require_company_permissions(
        user,
        country_name,
        city_name,
        type_name,
        (country, city, client_type, department),
    )
    if country_name and country is None:
        url_name = slugify(country_name)[:50]
        if not url_name:
            raise ValidationError(
                "The country name cannot be converted to a URL slug.")
        country = Country.objects.create(name=country_name, url_name=url_name)
    if city_name and city is None:
        city = City.objects.create(name=city_name, country=country)
    if type_name and client_type is None:
        client_type = ClientType.objects.create(
            name=type_name,
            department=department,
        )
    return country, city, client_type, department


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


class _DuplicateCompanyError(Exception):
    pass


def _format_schema_entry(name: str, entry: dict, is_matched: bool) -> str:
    """Render one model entry compactly. Matched models include fields;
    expanded models only include relationship lines to save context."""
    verbose = entry.get("verbose_name", name)
    if verbose and verbose.lower() != name.lower():
        lines = [f"## {name} ({verbose})"]
    else:
        lines = [f"## {name}"]

    if is_matched:
        all_fields = entry.get("fields", [])
        fields = all_fields[:MAX_SCHEMA_FIELDS_PER_MODEL]
        if fields:
            field_lines = ", ".join(
                f"{f['name']}: {f['type']}" for f in fields
            )
            remaining = len(all_fields) - len(fields)
            if remaining > 0:
                field_lines += f" [+{remaining} more]"
            lines.append(f"Fields: {field_lines}")

    forward = entry.get("relationships", [])[
        :MAX_SCHEMA_RELATIONSHIPS_PER_MODEL
    ]
    reverse = entry.get("reverse_relationships", [])[
        :MAX_SCHEMA_RELATIONSHIPS_PER_MODEL
    ]

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


@tool
def prepare_lead(
    first_name: str,
    email: str,
    company_name: str = "",
    last_name: str = "",
    phone: str = "",
    *,
    config: RunnableConfig,
) -> str:
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

    user = _current_user(config)
    _require_permission(user, "crm.add_lead")

    # 1. Pause the graph and hand the data to the frontend
    user_decision = interrupt(
        _approval_payload(
            action="CREATE_LEAD",
            record_type="Lead",
            data={k: v for k, v in lead_data.items() if v},
        )
    )

    # 2. This line ONLY runs after the graph is resumed with Command(resume=...)
    if user_decision == "approved":
        user = _current_user(config)
        _require_permission(user, "crm.add_lead")
        try:
            # Check for duplicates first (case-insensitive)
            if Lead.objects.filter(email__iexact=lead_data["email"]).exists():
                return _cancelled(
                    f"A lead with the email {lead_data['email']} already exists in the CRM database."
                )

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
def prepare_company(
    full_name: str,
    country_name: str = "",
    city_name: str = "",
    type_name: str = "",
    phone: str = "",
    website: str = "",
    registration_number: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Prepare a company for creation after explicit user approval.

    Provide country, city, and company type by name when known. Existing related
    records are reused; missing supplied records are created after approval.
    Email mentioned in the conversation is not part of this company record.
    """
    company_data = {
        "full_name": full_name.strip(),
        "country_name": country_name.strip(),
        "city_name": city_name.strip(),
        "type_name": type_name.strip(),
        "phone": phone.strip(),
        "website": website.strip(),
        "registration_number": registration_number.strip(),
    }
    if not company_data["full_name"]:
        return _needs_details(
            "Ask the user for the company name before proceeding."
        )

    user = _current_user(config)
    try:
        references = _company_reference_state(
            user,
            company_data["country_name"],
            company_data["city_name"],
            company_data["type_name"],
        )
        for model, name, existing in (
            (Country, company_data["country_name"], references[0]),
            (City, company_data["city_name"], references[1]),
            (ClientType, company_data["type_name"], references[2]),
        ):
            if name and existing is None:
                max_length = model._meta.get_field("name").max_length
                if len(name) > max_length:
                    raise ValidationError(
                        f"{model._meta.verbose_name} names must be at most "
                        f"{max_length} characters."
                    )
        _require_company_permissions(
            user,
            company_data["country_name"],
            company_data["city_name"],
            company_data["type_name"],
            references,
        )
    except _CompanyDepartmentError as exc:
        return _error(f"{exc} The company was not created.")
    except ValidationError as exc:
        return _needs_details(
            " ".join(str(message) for message in exc.messages)
            + " Ask the user to clarify or correct the company details."
        )

    country = references[0]
    company = Company(
        full_name=company_data["full_name"],
        country=country,
        city=references[1],
        city_name=company_data["city_name"],
        type=references[2],
        phone=company_data["phone"],
        website=company_data["website"],
        registration_number=company_data["registration_number"],
        owner=user,
        department=references[3],
    )
    try:
        company.clean_fields(exclude={"email"})
    except ValidationError as exc:
        return _needs_details(
            " ".join(str(message) for message in exc.messages)
            + " Ask the user to correct the company details."
        )

    if (country is not None or not company_data["country_name"]) and (
        Company.objects.filter(
            full_name__iexact=company_data["full_name"],
            country=country,
        ).exists()
    ):
        return _cancelled(
            "A company with this name and country already exists."
        )

    pending = {
        key: value
        for key, value in company_data.items()
        if value
    }
    related_to_create = [
        name
        for name, record in (
            ("country", references[0]),
            ("city", references[1]),
            ("type", references[2]),
        )
        if company_data[f"{name}_name"] and record is None
    ]
    # Keep legacy key inside `data` so the current frontend keeps working,
    # and also expose it in the structured `related` block for Phase 9.
    if related_to_create:
        pending["related_records_to_create"] = related_to_create

    user_decision = interrupt(
        _approval_payload(
            action="CREATE_COMPANY",
            record_type="Company",
            data=pending,
            related={"to_create": related_to_create} if related_to_create else {},
        )
    )

    if user_decision != "approved":
        return _cancelled("The user rejected the creation of this company.")

    user = _current_user(config)
    try:
        with transaction.atomic():
            references = _get_or_create_company_references(
                user,
                company_data["country_name"],
                company_data["city_name"],
                company_data["type_name"],
            )
            if Company.objects.filter(
                full_name__iexact=company_data["full_name"],
                country=references[0],
            ).exists():
                raise _DuplicateCompanyError

            company = Company.objects.create(
                full_name=company_data["full_name"],
                country=references[0],
                city=references[1],
                city_name=company_data["city_name"],
                type=references[2],
                phone=company_data["phone"],
                website=company_data["website"],
                registration_number=company_data["registration_number"],
                owner=user,
                department=references[3],
            )
    except _DuplicateCompanyError:
        return _cancelled(
            "A company with this name and country already exists."
        )
    except _CompanyDepartmentError as exc:
        return _error(f"{exc} The company was not created.")
    except ValidationError as exc:
        return _needs_details(
            " ".join(str(message) for message in exc.messages)
            + " No company or related records were created. Ask the user to "
            "clarify or correct the company details."
        )
    return _ok(
        f"Company {company.full_name} created (ID: {company.pk})."
    )
    # ------------------------------------------------------------------
# Phase 4.2 — Payment read tools
# ------------------------------------------------------------------


PAYMENT_STATUS_MAP = {
    "received": Payment.RECEIVED,          # 'r'
    "r": Payment.RECEIVED,
    "guaranteed": Payment.GUARANTEED,      # 'g'
    "g": Payment.GUARANTEED,
    "high probability": Payment.HIGH_PROBABILITY,   # 'h'
    "h": Payment.HIGH_PROBABILITY,
    "low probability": Payment.LOW_PROBABILITY,     # 'l'
    "l": Payment.LOW_PROBABILITY,
}

MAX_PAYMENT_RESULTS = 25


def _normalise_payment_status(value: str) -> str | None:
    """Map user input ('received', 'r', ...) to a valid Payment.status code.
    Return None if unrecognised so callers can raise a clean error."""
    if not value:
        return None
    return PAYMENT_STATUS_MAP.get(value.strip().lower())


def _payment_to_dict(payment: Payment) -> dict:
    """Serialise a Payment without leaking Decimal or model objects into JSON."""
    deal = payment.deal
    company = deal.company if deal and deal.company_id else None
    return {
        "id": payment.pk,
        "amount": str(payment.amount),
        "currency": payment.currency.name if payment.currency_id else None,
        "status": payment.get_status_display(),
        "status_code": payment.status,
        "payment_date": payment.payment_date.isoformat() if payment.payment_date else None,
        "deal": {
            "id": deal.pk if deal else None,
            "name": deal.name if deal else None,
            "amount": str(deal.amount) if deal and deal.amount is not None else None,
        } if deal else None,
        "company": {
            "id": company.pk,
            "name": company.full_name,
        } if company else None,
        "contract_number": payment.contract_number or None,
        "invoice_number": payment.invoice_number or None,
        "order_number": payment.order_number or None,
        "through_representation": payment.through_representation,
    }


@tool
def search_payments(
    status: str = "",
    date_from: str = "",
    date_to: str = "",
    invoice_number: str = "",
    contract_number: str = "",
    order_number: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Search CRM payments by status, date range, or reference number.

    All parameters are optional. Dates use YYYY-MM-DD format.
    Returns up to 25 payment summaries. Use get_payment_details for full info.
    """
    _current_user(config)  # auth check only; no per-object filter here

    qs = Payment.objects.select_related(
        "deal", "deal__company", "currency"
    ).order_by("-payment_date", "-id")

    if status:
        status_code = _normalise_payment_status(status)
        if status_code is None:
            return _needs_details(
                f'Unknown payment status "{status}". '
                f"Valid options: received, guaranteed, high probability, "
                f"low probability."
            )
        qs = qs.filter(status=status_code)

    if date_from:
        try:
            qs = qs.filter(payment_date__gte=datetime.strptime(
                date_from, "%Y-%m-%d").date())
        except ValueError:
            return _needs_details(
                f'date_from "{date_from}" is not a valid YYYY-MM-DD date.'
            )
    if date_to:
        try:
            qs = qs.filter(payment_date__lte=datetime.strptime(
                date_to, "%Y-%m-%d").date())
        except ValueError:
            return _needs_details(
                f'date_to "{date_to}" is not a valid YYYY-MM-DD date.'
            )

    if invoice_number:
        qs = qs.filter(invoice_number__iexact=invoice_number.strip())
    if contract_number:
        qs = qs.filter(contract_number__iexact=contract_number.strip())
    if order_number:
        qs = qs.filter(order_number__iexact=order_number.strip())

    results = list(qs[:MAX_PAYMENT_RESULTS])
    if not results:
        return "No payments match the given criteria."

    payload = [_payment_to_dict(p) for p in results]
    return json.dumps(
        {"count": len(payload), "payments": payload},
        ensure_ascii=False,
    )


@tool
def get_payment_details(payment_id: int, *, config: RunnableConfig) -> str:
    """Return full details of a single payment by its numeric ID."""
    _current_user(config)

    payment = (
        Payment.objects.select_related("deal", "deal__company", "currency")
        .filter(pk=payment_id)
        .first()
    )
    if payment is None:
        return _needs_details(f"No payment found with ID {payment_id}.")

    return json.dumps(_payment_to_dict(payment), ensure_ascii=False)


@tool
def get_company_payments(
    company_name: str,
    status: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """List payments for a company (matched via Payment → Deal → Company).

    Company is matched by full_name (exact, case-insensitive).
    Optionally filter by status ('received', 'guaranteed', etc.).
    """
    _current_user(config)

    if not company_name.strip():
        return _needs_details("Provide a company name to search payments.")

    company = Company.objects.filter(
        full_name__iexact=company_name.strip()).first()
    if company is None:
        return _needs_details(
            f'No company found with name "{company_name}".'
        )

    qs = Payment.objects.select_related(
        "deal", "deal__company", "currency"
    ).filter(deal__company=company).order_by("-payment_date", "-id")

    if status:
        status_code = _normalise_payment_status(status)
        if status_code is None:
            return _needs_details(
                f'Unknown payment status "{status}".'
            )
        qs = qs.filter(status=status_code)

    results = list(qs[:MAX_PAYMENT_RESULTS])
    if not results:
        return f"No payments found for company {company.full_name}."

    payload = [_payment_to_dict(p) for p in results]
    return json.dumps(
        {
            "company": {"id": company.pk, "name": company.full_name},
            "count": len(payload),
            "payments": payload,
        },
        ensure_ascii=False,
    )


@tool
def get_deal_payments(deal_id: int, *, config: RunnableConfig) -> str:
    """List all payments attached to a specific Deal by its numeric ID."""
    _current_user(config)

    deal = Deal.objects.select_related("company").filter(pk=deal_id).first()
    if deal is None:
        return _needs_details(f"No deal found with ID {deal_id}.")

    qs = Payment.objects.select_related(
        "deal", "deal__company", "currency"
    ).filter(deal=deal).order_by("-payment_date", "-id")

    results = list(qs[:MAX_PAYMENT_RESULTS])
    if not results:
        return f"No payments found for deal #{deal.pk} ({deal.name})."

    payload = [_payment_to_dict(p) for p in results]
    return json.dumps(
        {
            "deal": {
                "id": deal.pk,
                "name": deal.name,
                "company": deal.company.full_name if deal.company_id else None,
            },
            "count": len(payload),
            "payments": payload,
        },
        ensure_ascii=False,
    )


@tool
def calculate_received_payments(
    company_name: str = "",
    deal_id: int | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """Sum received payments (status='received'). Provide either company_name
    or deal_id (or neither to total all received payments).

    Uses Django ORM aggregation with Decimal — the LLM must not compute totals.
    """
    _current_user(config)

    qs = Payment.objects.filter(status=Payment.RECEIVED)
    scope = "all received payments"

    if deal_id is not None:
        deal = Deal.objects.select_related(
            "company").filter(pk=deal_id).first()
        if deal is None:
            return _needs_details(f"No deal found with ID {deal_id}.")
        qs = qs.filter(deal=deal)
        scope = f"deal #{deal.pk} ({deal.name})"
    elif company_name:
        company = Company.objects.filter(
            full_name__iexact=company_name.strip()).first()
        if company is None:
            return _needs_details(
                f'No company found with name "{company_name}".'
            )
        qs = qs.filter(deal__company=company)
        scope = f"company {company.full_name}"

    totals = {}
    for row in qs.values("currency__name").annotate(
        total=models.Sum("amount")
    ):
        currency = row["currency__name"] or "UNKNOWN"
        totals[currency] = str(row["total"] or Decimal("0"))

    if not totals:
        return json.dumps(
            {"scope": scope, "count": 0, "totals_by_currency": {}},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "scope": scope,
            "count": qs.count(),
            "totals_by_currency": totals,
        },
        ensure_ascii=False,
    )

# ------------------------------------------------------------------
# Phase 4.3 — prepare_payment (write tool, Phase 3 contract)
# ------------------------------------------------------------------


def _parse_decimal(raw: str) -> Decimal | None:
    """Parse '500', '500.00', '$500' → Decimal. Return None on failure."""
    if raw is None:
        return None
    cleaned = str(raw).replace(",", "").replace(
        "$", "").replace("€", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _parse_iso_date(raw: str) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


@tool
def prepare_payment(
    deal_id: int,
    amount: str,
    currency_name: str = "",
    payment_date: str = "",
    status: str = "received",
    contract_number: str = "",
    invoice_number: str = "",
    order_number: str = "",
    through_representation: bool = False,
    *,
    config: RunnableConfig,
) -> str:
    """Prepare a Payment for creation, then ask for approval.

    Steps: resolve Deal + Currency, verify the deal's company if a company is
    implied, validate amount/status/date, request approval, then create the
    Payment inside a transaction on approval.

    Use this ONLY when the user explicitly wants to record a payment. Never
    invent values — the deal_id, amount, and currency must come from the user
    or from a prior read tool.
    """
    user = _current_user(config)
    _require_permission(user, "crm.add_payment")

    # ---- 1. Resolve deal (required) ----
    deal = Deal.objects.select_related("company", "currency").filter(
        pk=deal_id
    ).first()
    if deal is None:
        return _needs_details(
            f"No deal found with ID {deal_id}. Ask the user for a valid deal ID."
        )

    # ---- 2. Resolve currency (from argument, else from deal) ----
    resolved_currency = None
    currency_source = None
    if currency_name:
        resolved_currency = Currency.objects.filter(
            name__iexact=currency_name.strip()
        ).first()
        if resolved_currency is None:
            available = list(
                Currency.objects.values_list("name", flat=True)[:20]
            )
            return _needs_details(
                f'Currency "{currency_name}" does not exist. '
                f"Available currencies: {', '.join(available) or 'none'}."
            )
        currency_source = "user"
    elif deal.currency_id:
        resolved_currency = deal.currency
        currency_source = "deal"
    else:
        return _needs_details(
            "No currency supplied and the deal has none. "
            "Ask the user which currency to record the payment in."
        )

    # ---- 3. Validate amount ----
    amount_dec = _parse_decimal(amount)
    if amount_dec is None:
        return _needs_details(
            f'Amount "{amount}" is not a valid number. Ask the user to correct it.'
        )
    if amount_dec <= Decimal("0"):
        return _needs_details(
            "Amount must be greater than zero. Ask the user to correct it."
        )
    if amount_dec.as_tuple().exponent < -2:
        return _needs_details(
            "Amount has more than 2 decimal places. Round it and try again."
        )

    # ---- 4. Validate status ----
    status_code = _normalise_payment_status(status)
    if status_code is None:
        return _needs_details(
            f'Unknown payment status "{status}". '
            f"Valid options: received, guaranteed, high probability, "
            f"low probability."
        )

    # ---- 5. Validate payment date ----
    payment_date_obj = (
        _parse_iso_date(payment_date) if payment_date else date.today()
    )
    if payment_date_obj is None:
        return _needs_details(
            f'payment_date "{payment_date}" is not a valid YYYY-MM-DD date.'
        )

    # ---- 6. Build the approval payload ----
    related = {
        "deal": {
            "id": deal.pk,
            "name": deal.name,
        },
        "currency": {
            "name": resolved_currency.name,
            "source": currency_source,   # 'user' or 'deal'
        },
    }
    if deal.company_id:
        related["company"] = {
            "id": deal.company.pk,
            "name": deal.company.full_name,
        }

    proposed = {
        "amount": str(amount_dec),
        "currency": resolved_currency.name,
        "status": status_code,
        "status_label": dict(Payment.STATUS_CHOICES)[status_code],
        "payment_date": payment_date_obj.isoformat(),
        "contract_number": contract_number.strip() or None,
        "invoice_number": invoice_number.strip() or None,
        "order_number": order_number.strip() or None,
        "through_representation": bool(through_representation),
    }

    warnings = []
    if deal.company_id is None:
        warnings.append(
            "This deal has no company attached; the payment will not appear "
            "under any company."
        )

    # ---- 7. Ask for approval ----
    decision = interrupt(
        _approval_payload(
            action="CREATE_PAYMENT",
            record_type="Payment",
            data=proposed,
            related=related,
            warnings=warnings,
        )
    )

    if decision != "approved":
        return _cancelled("The user rejected the creation of this payment.")

    # ---- 8. Post-approval re-checks (permission may have changed) ----
    user = _current_user(config)
    _require_permission(user, "crm.add_payment")

    deal = Deal.objects.filter(pk=deal.pk).first()
    if deal is None:
        return _cancelled(
            f"Deal #{deal_id} no longer exists. Payment was not created."
        )

    resolved_currency = Currency.objects.filter(
        name=resolved_currency.name
    ).first()
    if resolved_currency is None:
        return _cancelled(
            "The selected currency no longer exists. Payment was not created."
        )

    # Re-validate amount in case model rules tightened
    if amount_dec <= Decimal("0"):
        return _cancelled("Amount is not positive. Payment was not created.")

    # ---- 9. Create inside a transaction ----
    try:
        with transaction.atomic():
            payment = Payment.objects.create(
                deal=deal,
                amount=amount_dec,
                currency=resolved_currency,
                payment_date=payment_date_obj,
                status=status_code,
                contract_number=contract_number.strip() or "",
                invoice_number=invoice_number.strip() or "",
                order_number=order_number.strip() or "",
                through_representation=bool(through_representation),
            )
    except ValidationError as exc:
        return _cancelled(
            "Payment failed model validation: "
            + " ".join(str(m) for m in exc.messages)
        )
    except Exception as exc:
        return _error(f"Could not create payment. Details: {exc}")

    return _ok(
        f"Payment #{payment.pk} created: "
        f"{payment.amount} {payment.currency.name} "
        f"on {payment.payment_date.isoformat()} "
        f"(status: {payment.get_status_display()}) for deal #{deal.pk}."
    )


tools = [
    search_database_schema,
    list_departments,
    prepare_lead,
    prepare_company,
    prepare_payment,
    search_payments,
    get_payment_details,
    get_company_payments,
    get_deal_payments,
    calculate_received_payments,
]
