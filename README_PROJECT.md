# Project README: Django-CRM and AI Assistant

This guide explains how the Django-CRM codebase is organized, how to run it,
and how the AI assistant added to this checkout works. It is intended for
developers and operators who are new to the project.

For the upstream CRM product overview, see the [main README](README.md).
For full CRM installation and business configuration instructions, see the
[installation and configuration guide](docs/installation_and_configuration_guide.md)
and the [user guide](docs/django-crm_user_guide.md).

## What this project does

Django-CRM is a self-hosted customer relationship management system built with
Django. Its primary interface is Django Admin, extended with CRM-specific
screens and workflows. The project includes customer and sales records, task
management, analytics, email and mailing tools, shared chat, reminders, and
other business utilities.

This checkout also includes an AI assistant. It can answer CRM schema
questions using a local search index, prepare leads, and create projects,
tasks, and subtasks with optional tags, assignees, and reminders. Each write
pauses for explicit approval before changing the database.
The assistant is an addition to the CRM; it does not replace the regular
admin interface or the CRM's existing permissions and business workflows.

## Main components

| Component | Purpose |
| --- | --- |
| `crm/` | Core CRM records and business workflows, including leads, companies, contacts, deals, requests, products, and payments. |
| `tasks/` | Tasks, projects, memos, and related collaboration features. |
| `analytics/` | Reporting and analytics views. |
| `massmail/` | Email accounts, message handling, and campaign/mailing features. |
| `chat/` | Shared chat functionality used by CRM and task objects. |
| `common/` | Shared models, views, utilities, templates, static assets, and Django Admin customizations. |
| `help/`, `quality/`, `settings/`, `voip/` | Help content, quality-related features, site settings, and VoIP integrations. |
| `ai_assistant/` | Gemini-first/Groq-fallback chat agent, schema search, lead and task-management tools, approval UI, and conversation checkpointing. |
| `webcrm/` | Django project settings, root URL configuration, and application entry points. |
| `templates/`, `static/`, `media/` | Shared templates, source static assets, and user-uploaded/runtime media. |
| `tests/` | Tests for the CRM applications and workflows. |
| `docs/` | Installation, user, feature, and developer documentation. |

### Django request flow

1. `manage.py` loads `webcrm.settings` and starts Django management commands.
2. `webcrm/urls.py` routes requests to the CRM applications, Django Admin, and
   the AI assistant.
3. Django applications provide models, views, admin classes, templates, and
   migrations for their features.
4. Django ORM reads and writes the configured database. The default local
   configuration uses SQLite.

The CRM and admin URLs use configurable prefixes from `webcrm/settings.py`
(`SECRET_CRM_PREFIX` and `SECRET_ADMIN_PREFIX`) and Django's language URL
prefixes. For example, with the checked-in development settings, the admin
login is under `/en/456-admin/`. Do not assume these example prefixes will
remain the same if settings are changed.

## AI assistant

### Components and files

| Path | Responsibility |
| --- | --- |
| `ai_assistant/views.py` | Renders the chat page using the Django Admin base template; the page view is staff-member protected. |
| `ai_assistant/templates/ai_assistant/chat.html` | Chat layout, approval controls, and basic Markdown rendering. |
| `ai_assistant/urls.py` | Maps the chat page, chat API, and approval API. |
| `ai_assistant/api_views.py` | Accepts chat requests and resumes approved/rejected conversations. |
| `ai_assistant/agent.py` | Configures the LangGraph workflow and SQLite checkpoint saver. |
| `ai_assistant/llm.py` | Configures Gemini-first model invocation and Groq fallback with tools bound to both. |
| `ai_assistant/tools.py` | Implements schema lookup, lead preparation, and central tool registration. |
| `ai_assistant/task_tools.py` | Implements read-only project/team/task lookup and approval-gated project/task/subtask creation. |
| `ai_assistant/rag/` | Embeds schema descriptions and searches a FAISS vector index. |
| `ai_assistant/data/schema_catalog.json` | Extracted CRM model field names, types, and required flags. |
| `ai_assistant/data/index_metadata.json` and `vector.index` | Maps indexed vectors to model names and stores the FAISS index. |
| `ai_assistant/tests.py` | Focused tests for schema results and the created-lead link. |
| `checkpoints.sqlite` | LangGraph conversation state and interrupted approvals, created at runtime. |

### Chat and tool workflow

1. The chat page sends a message and a conversation `thread_id` to
   `POST /api/ai/chat/`.
2. The LangGraph agent sends the conversation to Gemini first and falls back
   to Groq if Gemini raises an error. Both models receive the same registered
   tools. The system prompt directs schema questions to the schema tool, lead
   requests to the lead tool, and project/task requests through the staged
   Project → Task → Subtask workflow.
3. For a schema question, the tool embeds the query, searches the CRM schema
   index, and returns the highest-ranked model with at most 10 field names and
   types. Keeping the response compact limits irrelevant context; semantic
   search can still select the wrong model, so verify important answers
   against the CRM code or database.
4. For project/task work, read-only tools resolve project names, task IDs,
   and team-member usernames. The agent calls `prepare_project`,
   `prepare_task`, and `prepare_subtask` separately and in hierarchy order.
   The tools support tags, assignees, and reminders.
5. Every `prepare_*` write tool pauses the graph and returns its proposed data
   to the UI. The user approves or rejects each proposal; each approval can
   resume the graph into another pending approval before the workflow ends.
6. On approval, the requested CRM object and associations are saved. A
   rejection cancels that proposed write without creating it.

The browser renders bold text, simple ordered/unordered lists, and links in
assistant messages. It builds DOM elements rather than inserting arbitrary
assistant HTML, and only permits HTTP(S) link targets. This is a deliberately
small Markdown subset, not a complete Markdown implementation.

### Conversation context and persistence

Each conversation uses its own `thread_id`. LangGraph's SQLite saver persists
graph state and paused approval steps in `checkpoints.sqlite`. A follow-up
request must reuse the same thread ID to continue that conversation. Before
each provider call, the agent retains at most the four most recent user turns
and drops older complete turns when their serialized history exceeds 6,000
characters. Schema tool results are limited to one model, at most 10
name-and-type field entries, and 1,000 characters; model completions are
limited to 512 tokens.

The SQLite checkpoint file still retains conversation state; treat it as
application data, protect it from public access, and include it in
backup/retention planning as appropriate.

To inspect the exact LangChain messages passed to either provider at each agent step,
set `AI_ASSISTANT_LOG_PROMPTS=1` before starting Django. The serialized
messages, including the system prompt, conversation history, and tool results,
are printed to the server console immediately before each model call. Prompt
traces can contain sensitive CRM and user data; enable this only in a
controlled development environment and protect or discard the resulting logs
appropriately.

### Refreshing the schema index

The checked-in schema catalog and FAISS index are generated from CRM models.
After changing CRM models, rebuild both artifacts from the repository root:

```powershell
python ai_assistant\run_extractor.py
python ai_assistant\rag\build_index.py
```

The index builder uses the `BAAI/bge-small-en-v1.5` embedding model and may
download model files the first time it runs. The assistant search code reads
the catalog, metadata, and vector index from `ai_assistant/data/`.

## Set up a local development environment

The upstream installation guide contains the full CRM setup instructions.
For a typical local SQLite development setup:

1. Install a supported Python version. The main README lists Python 3.12+.
2. From the repository root, create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the base project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **AI assistant dependencies:** the current base `requirements.txt` does
   not list the AI stack. To use the assistant, the environment also needs
   compatible versions of `langchain-google-genai`, `langchain-groq`,
   `langchain-core`, `langgraph`,
   `langgraph-checkpoint-sqlite`, `python-dotenv`, `fastembed`, `faiss-cpu`,
   `numpy`, and `djangorestframework`. Pin and add the versions tested by
   your deployment to its dependency manifest so another developer can
   reproduce the environment.
5. Add `GEMINI_API_KEY` and `GROQ_API_KEY` values to a local `.env` file in
   the project root. Gemini (`gemini-3.8-flash`) is tried first and Groq is
   used automatically if Gemini fails. `.env` is ignored by Git; do not
   commit API keys.
6. Apply migrations and create a login:

   ```powershell
   python manage.py migrate
   python manage.py createsuperuser
   ```

7. Start the development server:

   ```powershell
   python manage.py runserver
   ```

The default settings use SQLite, `DEBUG = True`, and local host names. They
are for local development, not public deployment. Configure secrets,
`ALLOWED_HOSTS`, database, email, HTTPS, static files, and other security
settings before deploying.

### Open the CRM and assistant

With the checked-in settings and English locale:

| Page or endpoint | URL |
| --- | --- |
| Django Admin login | `http://127.0.0.1:8000/en/456-admin/` |
| AI assistant page | `http://127.0.0.1:8000/api/ai/chat-ui/` |
| AI chat API | `POST http://127.0.0.1:8000/api/ai/chat/` |
| AI approval API | `POST http://127.0.0.1:8000/api/ai/approve/` |

The chat page requires a staff login. The chat and approval API views currently
disable DRF authentication and permissions and are CSRF-exempt. **Do not expose
these API endpoints to an untrusted network as-is.** Add and test appropriate
authentication, authorization, CSRF protections, input validation, and
request limits before production use. The assistant's prompt instructions and
approval interaction are not substitutes for endpoint security.

The development server serves static files for local work. For a production
deployment, follow Django's deployment/static-files guidance rather than
relying on development static serving.

## Tests and checks

Run Django's configuration checks and focused assistant tests from the
repository root:

```powershell
python manage.py check
python manage.py test ai_assistant.tests
```

The project also has a broader suite under `tests/`; run it with
`python manage.py test` when making wider changes.

## Further reading

- [Main project README](README.md) — product overview and feature summary.
- [Installation and configuration](docs/installation_and_configuration_guide.md) —
  CRM setup, email, database, users, and integrations.
- [User guide](docs/django-crm_user_guide.md) — using the CRM.
- [System overview](docs/crm_system_overview.md) — CRM concepts and workflows.
- [Changelog](CHANGELOG.md) — upstream project changes.
