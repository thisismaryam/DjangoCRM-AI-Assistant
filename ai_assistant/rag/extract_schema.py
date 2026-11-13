from django.apps import apps
import django
import os
import sys
import json
from pathlib import Path

# --------------------------------------------------
# 1. Initialize Django
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "webcrm.settings"
)

django.setup()


# --------------------------------------------------
# 2. Extract CRM models
# --------------------------------------------------

def get_relationship_type(field):

    if field.many_to_many:
        return "ManyToMany"

    if field.one_to_one:
        return "OneToOne"

    if field.many_to_one:
        return "ForeignKey"

    return field.__class__.__name__


def extract_models():

    schema_catalog = {}

    crm_app = apps.get_app_config("crm")

    models = list(crm_app.get_models())

    # Prepare entries for every model first
    for model in models:

        model_name = model.__name__
        meta = model._meta

        fields_info = []
        relationships = []

        # Only forward fields: exclude auto-created reverse relations
        for field in meta.get_fields():
            # Do not duplicate their reverse relationships.
            if meta.proxy:
                continue

            if field.auto_created:
                continue

            field_info = {
                "name": field.name,
                "type": field.get_internal_type(),
                "null": field.null,
                "blank": field.blank,
                "has_default": field.has_default(),
                "required_for_input": (
                    not field.blank
                    and not field.has_default()
                    and not getattr(field, "auto_now", False)
                    and not getattr(field, "auto_now_add", False)
                )
            }

            # Capture relationships
            if field.is_relation and field.related_model:

                related_model = field.related_model

                relationship = {
                    "field": field.name,
                    "type": get_relationship_type(field),
                    "related_model": related_model.__name__,
                    "related_model_label": related_model._meta.label,
                    "required_for_input": (
                        not field.blank
                        and not field.has_default()
                        if not field.many_to_many
                        else not field.blank
                    )
                }

                if field.remote_field:
                    related_name = getattr(
                        field.remote_field,
                        "related_name",
                        None
                    )

                    if related_name:
                        relationship["related_name"] = related_name

                    on_delete = getattr(
                        field.remote_field,
                        "on_delete",
                        None
                    )

                    if on_delete:
                        relationship["on_delete"] = getattr(
                            on_delete,
                            "__name__",
                            str(on_delete)
                        )

                relationships.append(relationship)

            fields_info.append(field_info)

        schema_catalog[model_name] = {
            "model": model_name,
            "app_label": meta.app_label,
            "model_label": meta.label,
            "verbose_name": str(meta.verbose_name),
            "proxy": meta.proxy,
            "concrete_model": meta.concrete_model.__name__,
            "fields": fields_info,
            "relationships": relationships,
            "reverse_relationships": []
        }

    # --------------------------------------------------
    # 2b. Record proxy model names on their concrete models
    # --------------------------------------------------

    for model in models:

        if model._meta.proxy:

            concrete_name = model._meta.concrete_model.__name__

            if concrete_name in schema_catalog:
                schema_catalog[concrete_name].setdefault(
                    "proxy_models", []
                ).append(model.__name__)

    # --------------------------------------------------
    # 3. Build reverse relationships
    # --------------------------------------------------

    for model in models:

        source_name = model.__name__
        source_meta = model._meta

        # Proxy models share the concrete model's database table.
        # Do not duplicate their reverse relationships.
        if source_meta.proxy:
            continue

        for field in source_meta.get_fields():

            if field.auto_created:
                continue

            if not field.is_relation or not field.related_model:
                continue

            target_name = field.related_model.__name__

            if target_name not in schema_catalog:
                continue

            reverse_name = field.remote_field.get_accessor_name()

            schema_catalog[target_name][
                "reverse_relationships"
            ].append({
                "related_model": source_name,
                "related_model_label": source_meta.label,
                "field": field.name,
                "accessor_name": reverse_name,
                "type": get_relationship_type(field)
            })

    # --------------------------------------------------
    # 4. Create searchable descriptions
    # --------------------------------------------------

    for model_name, data in schema_catalog.items():

        field_text = ", ".join(
            f"{field['name']} ({field['type']})"
            for field in data["fields"]
        )

        relationship_text = ", ".join(
            f"{rel['field']} connects to {rel['related_model']} "
            f"through {rel['type']}"
            for rel in data["relationships"]
        )

        reverse_text = ", ".join(
            f"{rel['related_model']} connects to this model "
            f"through {rel['field']}"
            for rel in data["reverse_relationships"]
        )

        description = (
            f"{model_name} is a CRM model. "
            f"Fields: {field_text}. "
            f"Relationships: {relationship_text}. "
            f"Related models: {reverse_text}."
        )

        data["description"] = description

    return schema_catalog


# --------------------------------------------------
# 5. Save schema catalog
# --------------------------------------------------

if __name__ == "__main__":

    print("Extracting relationship-aware CRM schema...")

    catalog = extract_models()

    data_dir = PROJECT_ROOT / "ai_assistant" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "schema_catalog.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=4)

    print(f"\nSUCCESS: Extracted {len(catalog)} models.")
    print(f"Saved to: {output_path}")

    for name, data in catalog.items():

        proxy_note = ""
        if data.get("proxy_models"):
            proxy_note = f", proxy_models={data['proxy_models']}"

        print(
            f"{name}: "
            f"{len(data['fields'])} fields, "
            f"{len(data['relationships'])} forward relationships, "
            f"{len(data['reverse_relationships'])} reverse relationships"
            f"{proxy_note}"
        )
