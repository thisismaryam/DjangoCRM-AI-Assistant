import json
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.utils.text import slugify
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ai_assistant.rag.search import search_schema
from common.models import Department
from crm.models import City, ClientType, Company, Country, Lead


SCHEMA_CATALOG_PATH = Path(__file__).resolve().parent / \
    "data" / "schema_catalog.json"
MAX_SCHEMA_FIELDS = 10
MAX_SCHEMA_RESULT_CHARS = 1000

with SCHEMA_CATALOG_PATH.open(encoding="utf-8") as schema_file:
    SCHEMA_CATALOG = json.load(schema_file)


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
    departments = Department.objects.filter(user=user).distinct().order_by("name")
    return json.dumps(
        [
            {"id": department.pk, "name": department.name}
            for department in departments
        ],
        ensure_ascii=False,
    )


class _DuplicateCompanyError(Exception):
    pass


@tool
def search_database_schema(query: str) -> str:
    """Useful for when you need to know which database model or table stores specific information."""
    results = search_schema(query, top_k=1)
    if not results:
        return "No matching database model was found."

    model_name = results[0]["model"]
    lines = [f"Model: {model_name}", "Fields:"]
    for field in SCHEMA_CATALOG[model_name]["fields"][:MAX_SCHEMA_FIELDS]:
        field_line = f"- {field['name']}: {field['type']}"
        if len("\n".join([*lines, field_line])) > MAX_SCHEMA_RESULT_CHARS:
            omitted_line = "- Additional fields omitted to fit the context limit."
            if len("\n".join([*lines, omitted_line])) <= MAX_SCHEMA_RESULT_CHARS:
                lines.append(omitted_line)
            break
        lines.append(field_line)

    return "\n".join(lines)


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
    user_decision = interrupt({
        "action": "CREATE_LEAD",
        "data": {k: v for k, v in lead_data.items() if v}
    })

    # 2. This line ONLY runs after the graph is resumed with Command(resume=...)
    if user_decision == "approved":
        user = _current_user(config)
        _require_permission(user, "crm.add_lead")
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
                f"<a href='/en/456-admin/crm/lead/{new_lead.id}/change/' "
                f"target='_blank'>Click here to view the Lead in CRM</a>."
            )

        except Exception as e:
            return f"ERROR: Could not save lead to database. Details: {str(e)}"
    else:
        return "CANCELLED: The user rejected the creation of this lead."


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
        return "NEEDS_DETAILS: Ask the user for the company name before proceeding."

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
        return f"ERROR: {exc} The company was not created."
    except ValidationError as exc:
        return (
            "NEEDS_DETAILS: "
            + " ".join(str(message) for message in exc.messages)
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
        return (
            "NEEDS_DETAILS: "
            + " ".join(str(message) for message in exc.messages)
            + " Ask the user to correct the company details."
        )

    if (country is not None or not company_data["country_name"]) and (
        Company.objects.filter(
            full_name__iexact=company_data["full_name"],
            country=country,
        ).exists()
    ):
        return "CANCELLED: A company with this name and country already exists."

    pending = {
        key: value
        for key, value in company_data.items()
        if value
    }
    pending["related_records_to_create"] = [
        name
        for name, record in (
            ("country", references[0]),
            ("city", references[1]),
            ("type", references[2]),
        )
        if company_data[f"{name}_name"] and record is None
    ]
    user_decision = interrupt({
        "action": "CREATE_COMPANY",
        "data": pending,
    })

    if user_decision != "approved":
        return "CANCELLED: The user rejected the creation of this company."

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
        return "CANCELLED: A company with this name and country already exists."
    except _CompanyDepartmentError as exc:
        return f"ERROR: {exc} The company was not created."
    except ValidationError as exc:
        return (
            "NEEDS_DETAILS: "
            + " ".join(str(message) for message in exc.messages)
            + " No company or related records were created. Ask the user to "
            "clarify or correct the company details."
        )
    return (
        f"SUCCESS: Company {company.full_name} created "
        f"(ID: {company.pk})."
    )


tools = [search_database_schema, prepare_lead, prepare_company, list_departments]
