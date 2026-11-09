import os
import json
from django.apps import apps
from django.core.management.base import BaseCommand


def extract_models():
    schema_catalog = {}

    # We only want to extract models from the 'crm' app (Leads, Companies, etc.)
    # If you want other apps like 'tasks', add them to this list.
    target_apps = ['crm']

    for app_label in target_apps:
        app_config = apps.get_app_config(app_label)

        for model in app_config.get_models():
            model_name = model.__name__
            verbose_name = str(model._meta.verbose_name)

            # Gather all fields, including inherited ones
            fields_info = []
            for field in model._meta.get_fields():
                # Skip reverse relations (like the lists of deals attached to a company)
                if not field.is_relation or field.many_to_one or field.one_to_one:
                    field_type = field.get_internal_type()
                    is_required = not field.blank and not field.null

                    fields_info.append({
                        "name": field.name,
                        "type": field_type,
                        "required": is_required
                    })

            # Create a text description for the RAG to embed later
            description = f"The {model_name} model ({verbose_name}) has the following fields: "
            description += ", ".join(
                [f"{f['name']} ({f['type']})" for f in fields_info])

            schema_catalog[model_name] = {
                "description": description,
                "fields": fields_info
            }

    return schema_catalog


if __name__ == "__main__":
    # This block allows us to run the script easily
    print("Extracting schema...")
    catalog = extract_models()

    # Ensure the data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save the JSON
    output_path = os.path.join(data_dir, 'schema_catalog.json')
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=4)

    print(f"Successfully saved {len(catalog)} models to {output_path}")
