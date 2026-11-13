import uuid

from django.core.exceptions import (
    ObjectDoesNotExist,
    PermissionDenied,
    ValidationError,
)
from django.db import DatabaseError, IntegrityError
from django.db.utils import DataError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ai_assistant.agent import app


def _request_config(user, public_thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": f"user-{user.pk}:{public_thread_id}",
            "user_id": user.pk,
        }
    }


def _get_public_thread_id(data: dict) -> str | None:
    thread_id = data.get("thread_id")
    if thread_id is None:
        return str(uuid.uuid4())
    if not isinstance(thread_id, str) or not thread_id or len(thread_id) > 128:
        return None
    return thread_id


def _snapshot_interrupt(snapshot) -> dict | None:
    """Return the pending interrupt payload from the persisted graph state."""
    if not snapshot:
        return None
    for task in snapshot.tasks or ():
        for pending in getattr(task, "interrupts", ()) or ():
            value = getattr(pending, "value", None)
            if value is not None:
                return value
    return None


def _last_ai_text(result: dict) -> str:
    """Return the last AI message text, ignoring tool-call-only messages."""
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Tool result — keep going up the stack to find the AI response after it
            continue
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                # This AI message only requested a tool; skip it
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(
                        part, dict) else str(part)
                    for part in content
                )
            if content and content.strip():
                return content
    return ""


def _build_response(config: dict, thread_id: str, result: dict, decision: str | None = None):
    # The invoke return value can be incomplete when a node pauses. The state
    # snapshot is authoritative for both pending approvals and completed turns.
    snapshot = app.get_state(config)
    payload = _snapshot_interrupt(snapshot)
    if payload is not None:
        return Response({
            "status": "pending_approval",
            "thread_id": thread_id,
            "payload": payload,
        })
    state = getattr(snapshot, "values", None) or result or {}
    text = _last_ai_text(state)
    if not text:
        text = "Decision processed." if decision else "I have processed your request."
    return Response({
        "status": "success",
        "thread_id": thread_id,
        "message": text,
    })


@api_view(["POST"])
@authentication_classes([SessionAuthentication])
@permission_classes([IsAuthenticated])
def chat_view(request):
    data = request.data
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return Response(
            {"status": "error", "message": "A non-empty message is required."},
            status=400,
        )
    thread_id = _get_public_thread_id(data)
    if thread_id is None:
        return Response(
            {"status": "error", "message": "Invalid thread_id."},
            status=400,
        )

    config = _request_config(request.user, thread_id)
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return _build_response(config, thread_id, result)
    except PermissionDenied:
        return Response(
            {"status": "error", "message": "You are not permitted to perform this action."},
            status=403,
        )
    except (IntegrityError, DataError, ValidationError, ObjectDoesNotExist):
        return Response({
            "status": "error",
            "message": "The request contains invalid or unavailable CRM data.",
        }, status=400)
    except DatabaseError:
        return Response({
            "status": "error",
            "message": "The CRM database could not complete the request.",
        }, status=503)
    except Exception as exc:
        return Response({"status": "error", "message": str(exc)}, status=500)


@api_view(["POST"])
@authentication_classes([SessionAuthentication])
@permission_classes([IsAuthenticated])
def approve_view(request):
    data = request.data
    thread_id = _get_public_thread_id(data)
    decision = data.get("decision")
    if thread_id is None or "thread_id" not in data:
        return Response(
            {"status": "error", "message": "A valid thread_id is required."},
            status=400,
        )
    if decision not in ("approved", "rejected"):
        return Response(
            {"status": "error", "message": "decision must be approved or rejected."},
            status=400,
        )

    config = _request_config(request.user, thread_id)
    try:
        snapshot = app.get_state(config)
        if not snapshot.values or not snapshot.values.get("messages"):
            return Response(
                {"status": "error", "message": "Conversation not found."},
                status=404,
            )
        result = app.invoke(Command(resume=decision), config=config)
        return _build_response(config, thread_id, result, decision)
    except PermissionDenied:
        return Response(
            {"status": "error", "message": "You are not permitted to perform this action."},
            status=403,
        )
    except (IntegrityError, DataError, ValidationError, ObjectDoesNotExist):
        return Response({
            "status": "error",
            "message": "Approval could not be completed because the CRM data is invalid or unavailable.",
        }, status=400)
    except DatabaseError:
        return Response({
            "status": "error",
            "message": "The CRM database could not complete the approval.",
        }, status=503)
    except Exception as exc:
        return Response({"status": "error", "message": str(exc)}, status=500)
