from django.apps import apps
import os
import sys
import json
import django

# Tell Python where Django settings are
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webcrm.settings')
django.setup()


print("Extracting models...")
schema_catalog = {}

# We only want the crm models
crm_app = apps.get_app_config('crm')

for model in crm_app.get_models():
    model_name = model.__name__
    verbose_name = str(model._meta.verbose_name)

    fields_info = []
    for field in model._meta.get_fields():
        if not field.is_relation or field.many_to_one or field.one_to_one:
            fields_info.append({
                "name": field.name,
                "type": field.get_internal_type(),
                "required": not field.blank and not field.null
            })

    description = f"The {model_name} model ({verbose_name}) has the following fields: "
    description += ", ".join([f"{f['name']} ({f['type']})" for f in fields_info])

    schema_catalog[model_name] = {
        "description": description,
        "fields": fields_info
    }

# Save it to a json file
data_dir = os.path.join('ai_assistant', 'data')
os.makedirs(data_dir, exist_ok=True)
output_path = os.path.join(data_dir, 'schema_catalog.json')

with open(output_path, 'w') as f:
    json.dump(schema_catalog, f, indent=4)

print(f"SUCCESS! Saved {len(schema_catalog)} models to {output_path}")
``
