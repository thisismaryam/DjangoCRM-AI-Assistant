import json
from datetime import date, datetime, time, timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from common.models import Reminder
from tasks.models import Project, ProjectStage, Tag, Task, TaskStage


MAX_LOOKUP_RESULTS = 50


def _current_user(config: RunnableConfig) -> Any:
    user_id = config.get("configurable", {}).get("user_id")
    if user_id is None:
        raise PermissionDenied("An authenticated user is required for this tool.")
    user = get_user_model().objects.filter(pk=user_id, is_active=True).first()
    if user is None:
        raise PermissionDenied("The authenticated user is no longer active.")
    return user


def _require_permission(user: Any, permission: str) -> None:
    if not user.has_perm(permission):
        raise PermissionDenied(f"Permission required: {permission}.")


def _visible_projects(user: Any):
    _require_permission(user, "tasks.view_project")
    projects = Project.objects.all()
    if user.is_superuser:
        return projects
    return projects.filter(
        Q(owner=user)
        | Q(co_owner=user)
        | Q(responsible=user)
        | Q(subscribers=user)
    ).distinct()


def _visible_tasks(user: Any):
    _require_permission(user, "tasks.view_task")
    tasks = Task.objects.all()
    if user.is_superuser:
        return tasks
    return tasks.filter(
        Q(owner=user)
        | Q(co_owner=user)
        | Q(responsible=user)
        | Q(subscribers=user)
        | Q(task__owner=user)
        | Q(task__responsible=user)
        | Q(task__subscribers=user)
        | Q(project__owner=user)
        | Q(project__responsible=user)
        | Q(project__subscribers=user)
    ).distinct()


def _assignable_users(user: Any, parent_task: Task | None = None):
    user_model = get_user_model()
    active_users = user_model.objects.filter(is_active=True)
    if any(
        getattr(user, role, False)
        for role in ("is_chief", "is_superuser", "is_task_operator")
    ):
        users = active_users
    elif getattr(user, "is_department_head", False):
        department_id = user.department_id
        if getattr(user, "is_accountant", False):
            department_id = (
                user.groups.filter(name="Bookkeeping")
                .values_list("id", flat=True)
                .first()
            )
        users = (
            active_users.filter(groups=department_id)
            if department_id
            else active_users.none()
        )
    else:
        users = active_users.filter(pk=user.pk)

    if parent_task is not None:
        users = (
            users
            | active_users.filter(
                pk__in=parent_task.responsible.values_list("pk", flat=True)
            )
        ).distinct()
    return users.order_by("username")


def _parse_iso_date(value: str, field_name: str) -> date | None:
    if not value:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.")
    return parsed


def _validate_dates(start_date: date | None, due_date: date | None) -> None:
    if start_date and due_date and due_date < start_date:
        raise ValueError("due_date cannot be earlier than start_date.")


def _resolve_assignees(
    user: Any,
    assignee_ids: list[int] | None,
    parent_task: Task | None = None,
) -> list[Any]:
    ids = list(dict.fromkeys(assignee_ids or []))
    if not ids:
        return []
    users = list(_assignable_users(user, parent_task).filter(pk__in=ids))
    found_ids = {user.pk for user in users}
    missing_ids = [user_id for user_id in ids if user_id not in found_ids]
    if missing_ids:
        raise ValueError(
            "Unknown or inactive assignee ID(s): "
            f"{missing_ids}. Resolve usernames with list_team_members first."
        )
    return users


def _apply_associations(
    instance: Project | Task,
    creator: Any,
    assignees: list[Any],
    tags: list[str] | None,
    reminder_date: date | None,
    reminder_description: str,
) -> None:
    if assignees:
        instance.responsible.set(assignees)

    content_type = ContentType.objects.get_for_model(instance)
    tag_names = dict.fromkeys(
        tag.strip() for tag in (tags or []) if tag.strip()
    )
    for tag_name in tag_names:
        tag, _ = Tag.objects.get_or_create(
            name=tag_name,
            for_content=content_type,
        )
        instance.tags.add(tag)

    if reminder_date:
        reminder_at = datetime.combine(reminder_date, time(hour=9))
        if settings.USE_TZ:
            reminder_at = timezone.make_aware(
                reminder_at,
                timezone.get_current_timezone(),
            )
        Reminder.objects.create(
            content_type=content_type,
            object_id=instance.pk,
            subject=instance.name,
            description=reminder_description,
            reminder_date=reminder_at,
            owner=creator,
        )


def _prepare_dates(
    start_date: str,
    due_date: str,
    reminder_date: str,
) -> tuple[date | None, date | None, date | None]:
    start = _parse_iso_date(start_date, "start_date")
    due = _parse_iso_date(due_date, "due_date")
    reminder = _parse_iso_date(reminder_date, "reminder_date")
    _validate_dates(start, due)
    return start, due, reminder


@tool
def list_projects(query: str = "", *, config: RunnableConfig) -> str:
    """Read-only lookup of projects by name. Call before prepare_task to resolve the exact project ID; this tool never changes data."""
    user = _current_user(config)
    projects = _visible_projects(user).select_related("stage").order_by("name")
    if query.strip():
        projects = projects.filter(name__icontains=query.strip())
    return json.dumps(
        [
            {
                "id": project.pk,
                "name": project.name,
                "stage": project.stage.name if project.stage else None,
            }
            for project in projects[:MAX_LOOKUP_RESULTS]
        ],
        ensure_ascii=False,
    )


@tool
def list_team_members(
    query: str = "",
    parent_task_id: int | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """Read-only lookup of active team members. Call before assigning anyone and use the returned user IDs; this tool never changes data."""
    user = _current_user(config)
    if not any(
        user.has_perm(permission)
        for permission in ("tasks.add_task", "tasks.add_project")
    ):
        raise PermissionDenied(
            "Permission required: tasks.add_task or tasks.add_project."
        )
    parent_task = None
    if parent_task_id is not None:
        parent_task = _visible_tasks(user).filter(pk=parent_task_id).first()
        if parent_task is None or parent_task.task_id is not None:
            return "Parent task was not found or is not accessible."
    members = _assignable_users(user, parent_task)
    if query.strip():
        search = query.strip()
        members = members.filter(
            Q(username__icontains=search)
            | Q(first_name__icontains=search)
            | Q(last_name__icontains=search)
        )
    return json.dumps(
        [
            {
                "id": member.pk,
                "username": member.get_username(),
                "name": member.get_full_name() or member.get_username(),
            }
            for member in members[:MAX_LOOKUP_RESULTS]
        ],
        ensure_ascii=False,
    )


@tool
def list_tasks(project_id: int, *, config: RunnableConfig) -> str:
    """Read-only lookup of tasks and subtasks in a project. Use IDs returned here to select existing tasks; this tool never changes data."""
    user = _current_user(config)
    if not _visible_projects(user).filter(pk=project_id).exists():
        return "Project was not found or is not accessible. Resolve it with list_projects."
    tasks = (
        _visible_tasks(user)
        .filter(project_id=project_id)
        .select_related("task")
        .order_by("id")
    )
    return json.dumps(
        [
            {
                "id": task.pk,
                "name": task.name,
                "parent_task_id": task.task_id,
                "parent_task_name": task.task.name if task.task else None,
            }
            for task in tasks[:MAX_LOOKUP_RESULTS]
        ],
        ensure_ascii=False,
    )


@tool
def prepare_project(
    name: str,
    description: str = "",
    start_date: str = "",
    due_date: str = "",
    priority: int = 2,
    next_step: str = "Review project progress",
    next_step_date: str = "",
    assignee_ids: list[int] | None = None,
    tags: list[str] | None = None,
    reminder_date: str = "",
    reminder_description: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Propose a project for human approval. All dates must be YYYY-MM-DD. Provide assignee_ids only after resolving usernames with list_team_members. This tool always pauses for approval before writing."""
    user = _current_user(config)
    _require_permission(user, "tasks.add_project")
    if not name.strip():
        return "ERROR: Project name is required."
    if priority not in (1, 2, 3):
        return "ERROR: priority must be 1 (low), 2 (middle), or 3 (high)."
    try:
        start, due, reminder = _prepare_dates(start_date, due_date, reminder_date)
        step_date = _parse_iso_date(next_step_date, "next_step_date")
        assignees = _resolve_assignees(user, assignee_ids)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if step_date is None:
        step_date = due or (timezone.localdate() + timedelta(days=1))
    data = {
        "name": name.strip(),
        "description": description,
        "start_date": start.isoformat() if start else None,
        "due_date": due.isoformat() if due else None,
        "priority": priority,
        "next_step": next_step,
        "next_step_date": step_date.isoformat(),
        "assignee_ids": [user.pk for user in assignees],
        "tags": tags or [],
        "reminder_date": reminder.isoformat() if reminder else None,
        "reminder_description": reminder_description,
    }
    if interrupt({"action": "CREATE_PROJECT", "data": data}) != "approved":
        return "CANCELLED: The user rejected project creation."

    user = _current_user(config)
    _require_permission(user, "tasks.add_project")
    assignees = _resolve_assignees(user, data["assignee_ids"])
    with transaction.atomic():
        project = Project.objects.create(
            owner=user,
            name=data["name"],
            description=description,
            start_date=start,
            due_date=due,
            priority=priority,
            next_step=next_step,
            next_step_date=step_date,
            stage=ProjectStage.objects.filter(default=True).first(),
        )
        _apply_associations(
            project, user, assignees, tags, reminder, reminder_description
        )
    return f"SUCCESS: Project '{project.name}' created with ID {project.pk}."


@tool
def prepare_task(
    project_id: int,
    name: str,
    description: str = "",
    start_date: str = "",
    due_date: str = "",
    priority: int = 2,
    next_step: str = "Review task progress",
    next_step_date: str = "",
    assignee_ids: list[int] | None = None,
    tags: list[str] | None = None,
    reminder_date: str = "",
    reminder_description: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Propose a top-level task under a project for human approval. First call list_projects and use its project ID. All dates must be YYYY-MM-DD. Resolve assignee IDs with list_team_members. This tool always pauses for approval before writing."""
    user = _current_user(config)
    _require_permission(user, "tasks.add_task")
    project = _visible_projects(user).filter(pk=project_id).first()
    if project is None:
        return "ERROR: Project was not found or is not accessible. Resolve it with list_projects."
    if not name.strip():
        return "ERROR: Task name is required."
    if priority not in (1, 2, 3):
        return "ERROR: priority must be 1 (low), 2 (middle), or 3 (high)."
    stage = TaskStage.objects.filter(default=True).first()
    if stage is None:
        return "ERROR: No default task stage is configured."
    try:
        start, due, reminder = _prepare_dates(start_date, due_date, reminder_date)
        step_date = _parse_iso_date(next_step_date, "next_step_date")
        assignees = _resolve_assignees(user, assignee_ids)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if step_date is None:
        step_date = due or (timezone.localdate() + timedelta(days=1))
    data = {
        "project_id": project.pk,
        "project_name": project.name,
        "name": name.strip(),
        "description": description,
        "start_date": start.isoformat() if start else None,
        "due_date": due.isoformat() if due else None,
        "priority": priority,
        "next_step": next_step,
        "next_step_date": step_date.isoformat(),
        "assignee_ids": [user.pk for user in assignees],
        "tags": tags or [],
        "reminder_date": reminder.isoformat() if reminder else None,
        "reminder_description": reminder_description,
    }
    if interrupt({"action": "CREATE_TASK", "data": data}) != "approved":
        return "CANCELLED: The user rejected task creation."

    user = _current_user(config)
    _require_permission(user, "tasks.add_task")
    project = _visible_projects(user).filter(pk=data["project_id"]).first()
    if project is None:
        raise PermissionDenied("The selected project is no longer accessible.")
    assignees = _resolve_assignees(user, data["assignee_ids"])
    with transaction.atomic():
        task = Task.objects.create(
            owner=user,
            project=project,
            name=data["name"],
            description=description,
            start_date=start,
            due_date=due,
            priority=priority,
            next_step=next_step,
            next_step_date=step_date,
            stage=stage,
        )
        _apply_associations(
            task, user, assignees, tags, reminder, reminder_description
        )
    return f"SUCCESS: Task '{task.name}' created with ID {task.pk} in project '{project.name}'."


@tool
def prepare_subtask(
    parent_task_id: int,
    name: str,
    description: str = "",
    start_date: str = "",
    due_date: str = "",
    priority: int = 2,
    next_step: str = "Review subtask progress",
    next_step_date: str = "",
    assignee_ids: list[int] | None = None,
    tags: list[str] | None = None,
    reminder_date: str = "",
    reminder_description: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Propose a subtask under an existing top-level project task for human approval. Use the parent task ID returned by prepare_task or list_tasks. All dates must be YYYY-MM-DD. Resolve assignee IDs with list_team_members. This tool always pauses for approval before writing."""
    user = _current_user(config)
    _require_permission(user, "tasks.add_task")
    parent = (
        _visible_tasks(user)
        .filter(pk=parent_task_id)
        .select_related("project")
        .first()
    )
    if parent is None or parent.project_id is None:
        return (
            f"ERROR: Project task ID {parent_task_id} was not found. "
            "Use list_tasks to resolve it."
        )
    if parent.task_id is not None:
        return (
            "ERROR: Subtasks cannot have nested subtasks; the supported "
            "hierarchy is Project -> Task -> Subtask."
        )
    if not name.strip():
        return "ERROR: Subtask name is required."
    if priority not in (1, 2, 3):
        return "ERROR: priority must be 1 (low), 2 (middle), or 3 (high)."
    stage = TaskStage.objects.filter(default=True).first()
    if stage is None:
        return "ERROR: No default task stage is configured."
    try:
        start, due, reminder = _prepare_dates(start_date, due_date, reminder_date)
        step_date = _parse_iso_date(next_step_date, "next_step_date")
        assignees = _resolve_assignees(user, assignee_ids, parent)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if step_date is None:
        step_date = due or (timezone.localdate() + timedelta(days=1))
    data = {
        "project_id": parent.project_id,
        "project_name": parent.project.name,
        "parent_task_id": parent.pk,
        "parent_task_name": parent.name,
        "name": name.strip(),
        "description": description,
        "start_date": start.isoformat() if start else None,
        "due_date": due.isoformat() if due else None,
        "priority": priority,
        "next_step": next_step,
        "next_step_date": step_date.isoformat(),
        "assignee_ids": [user.pk for user in assignees],
        "tags": tags or [],
        "reminder_date": reminder.isoformat() if reminder else None,
        "reminder_description": reminder_description,
    }
    if interrupt({"action": "CREATE_SUBTASK", "data": data}) != "approved":
        return "CANCELLED: The user rejected subtask creation."

    user = _current_user(config)
    _require_permission(user, "tasks.add_task")
    parent = (
        _visible_tasks(user)
        .filter(pk=data["parent_task_id"])
        .select_related("project")
        .first()
    )
    if parent is None or parent.project_id is None or parent.task_id is not None:
        raise PermissionDenied("The selected parent task is no longer accessible.")
    assignees = _resolve_assignees(user, data["assignee_ids"], parent)
    with transaction.atomic():
        subtask = Task.objects.create(
            owner=user,
            task=parent,
            project=parent.project,
            name=data["name"],
            description=description,
            start_date=start,
            due_date=due,
            priority=priority,
            next_step=next_step,
            next_step_date=step_date,
            stage=stage,
        )
        _apply_associations(
            subtask, user, assignees, tags, reminder, reminder_description
        )
    return (
        f"SUCCESS: Subtask '{subtask.name}' created with ID {subtask.pk} "
        f"under task '{parent.name}'."
    )


task_management_tools = [
    list_projects,
    list_team_members,
    list_tasks,
    prepare_project,
    prepare_task,
    prepare_subtask,
]
