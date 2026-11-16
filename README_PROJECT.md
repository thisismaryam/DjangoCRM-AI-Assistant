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
questions using a local search index, look up CRM records with database tools,
and prepare CRM records for creation. Write operations pause for explicit
approval before changing the database. A bundled workflow can prepare a
Company, Contact, and Deal together for one approval and create them in a
single transaction. The assistant is an addition to the CRM; it does not
replace the regular admin interface or the CRM's existing permissions and
business workflows.

##  Why this AI Assistant is different (Business Value)

Most AI integrations blindly pass data to a database, risking corrupted records and broken relationships. This assistant is built with **enterprise-grade safety rails**:

1. **Document-to-CRM Ingestion:** Users can upload a PDF or DOCX proposal. The AI extracts the Company, Contact, Deal value, Description, and Workflow, and maps it directly to the CRM schema.
2. **1-Click Bundles:** Creating a Company, Contact, and Deal usually takes 3 separate forms. Our AI uses `prepare_crm_bundle` to gather all three and present a **single unified approval card**. If one fails, the entire database transaction rolls back safely.
3. **Strict Relational Guardrails:** The AI verifies cross-model relationships. For example, it will *refuse* to create a Deal if the specified Contact person does not work for the specified Company.
4. **Human-in-the-Loop:** The AI can never write to the database silently. It pauses via LangGraph `interrupt()` and requires a human to click "Approve" on the UI before executing `transaction.atomic()`.
5. **Intelligent Context:** Using LangGraph state checkpointing, you can interrupt the AI mid-flow (e.g., *"Wait, change the deal amount to $7,500"*) and it will update the pending approval card without losing the rest of the data.


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
| `ai_assistant/` | CRM chat agent, schema search, document text extraction, CRM read/write tools, approval UI, and conversation checkpointing. |
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
| `ai_assistant/api_views.py` | Accepts text and document uploads, starts chat requests, and resumes approved/rejected conversations. |
| `ai_assistant/agent.py` | Defines the agent state, request-aware system prompt, conversation trimming, and LangGraph workflow. |
| `ai_assistant/context.py` | Keeps recent complete user turns within the configured conversation-history limits. |
| `ai_assistant/llm.py` | Configures Gemini primary and fallback models, followed by Groq, with CRM tools bound to each. |
| `ai_assistant/tools.py` | Implements schema lookup, CRM record reads, approval-gated write tools, the bundled Company/Contact/Deal workflow, and tool registration. |
| `ai_assistant/documents/extract.py` | Extracts text from PDF, DOCX, and TXT uploads. |
| `ai_assistant/rag/` | Embeds schema descriptions and searches a FAISS vector index. |
| `ai_assistant/data/schema_catalog.json` | Extracted CRM model field names, types, and required flags. |
| `ai_assistant/data/index_metadata.json` and `vector.index` | Maps indexed vectors to model names and stores the FAISS index. |
| `ai_assistant/tests.py` | Tests for schema search, CRM preparation and approval, bundle creation and rollback, provider fallback, and API behavior. |
| `checkpoints.sqlite` | LangGraph conversation state and interrupted approvals, created at runtime. |

### Chat and tool workflow

1. The chat page sends a message and conversation `thread_id` to
   `POST /api/ai/chat/`. It can also send a multipart upload in the `file`
   field, with or without a text message. Uploads are limited to 10 MB and
   support PDF, DOCX, and TXT files; extracted text is appended to the request.
2. The LangGraph agent uses Gemini primary, Gemini fallback, and then Groq if
   earlier providers raise errors. Each provider receives the same registered
   tools: schema search, department listing, CRM record reads, and approval-
   gated CRM write tools.
3. Schema questions use `search_database_schema`. Record questions and
   calculations use the relevant database read tool rather than relying on
   conversational memory. The schema index describes model structure; database
   tools verify actual records.
4. The write tools cover leads, companies, contacts, deals, and payments.
   They prepare a proposal and pause for explicit approval; rejecting a
   proposal leaves the database unchanged. Permissions are checked for writes.
   `prepare_company` can reuse existing Country, City, and ClientType records
   and create missing supplied references after approval. A city requires a
   country, and creating a missing ClientType requires exactly one department
   associated with the user.
5. For a request to create a Company, Contact, and Deal together, the agent
   calls `prepare_crm_bundle` instead of the individual write tools. One
   approval card summarizes all three records; existing matching Company and
   Contact records are reused. After approval, missing records are created in
   Company → Contact → Deal order in one database transaction, so a failed
   step rolls back the bundle.
6. Document extraction rules map an uploaded section titled “Workflow” or
   “Stages”, or a numbered list of steps, to the Deal's `workflow` field.
   General project-description text is mapped to the Deal's `description`
   field. These values are included in the bundle approval proposal.
7. The approval endpoint resumes only the current authenticated user's
   namespaced conversation. The assistant does not provide project or task
   management tools; those CRM features remain available through the regular
   application interfaces.

The browser renders bold text, simple ordered/unordered lists, and links in
assistant messages. It builds DOM elements rather than inserting arbitrary
assistant HTML, and only permits HTTP(S) link targets. This is a deliberately
small Markdown subset, not a complete Markdown implementation.

### Conversation context and persistence

Each conversation uses its own `thread_id`, namespaced by the authenticated
user on the server. An approval can only resume a thread in that user's
namespace. LangGraph's SQLite saver persists graph state and paused approval
steps in `checkpoints.sqlite`. A follow-up request must reuse the same public
thread ID to continue that conversation. Each chat request also supplies its
text in `original_request`, which the system prompt repeats separately from
trimmed message history. Before each provider call, the agent retains at most
the three most recent user turns (including their associated assistant and
tool messages) and drops older complete turns when serialized history exceeds
10,000 characters. Model completions are limited to 512 tokens.

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
   `langchain-core`, `langgraph`, `langgraph-checkpoint-sqlite`,
   `python-dotenv`, `fastembed`, `faiss-cpu`, `numpy`,
   `djangorestframework`, `pypdf`, and `python-docx`. Pin and add the versions
   tested by your deployment to its dependency manifest so another developer
   can reproduce the environment.
5. Add `GEMINI_API_KEY` and `GROQ_API_KEY` values to a local `.env` file in
   the project root. By default, the assistant tries Gemini
   (`gemini-2.5-flash`), then Gemini (`gemini-2.0-flash`), then Groq
   (`openai/gpt-oss-20b`) if an earlier provider fails. Model names can be
   overridden with the corresponding `GEMINI_MODEL`,
   `GEMINI_FALLBACK_MODEL`, and `GROQ_MODEL` variables. `.env` is ignored by
   Git; do not commit API keys.
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

The chat page and both API endpoints require an authenticated session. The
chat and approval APIs use Django REST Framework session authentication, so
unsafe requests must include a valid CSRF token. Approval requests are
restricted to the current user's namespaced conversation thread. Continue to
follow Django's deployment guidance and review production-specific
authentication, rate limiting, and operational requirements before exposing
the application publicly.

The development server serves static files for local work. For a production
deployment, follow Django's deployment/static-files guidance rather than
relying on development static serving.

## Tests and checks

Run Django's configuration checks and assistant tests from the repository
root. The assistant tests cover schema retrieval, CRM record preparation,
bundled approval and transaction rollback, provider fallback, tool
registration, document-extraction prompt guidance, and authenticated thread
isolation:

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
