import uuid
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# Import the compiled graph and the interrupt state
from ai_assistant.agent import app
import json


@csrf_exempt
@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def chat_view(request):
    try:
        data = json.loads(request.body)
        message = data.get('message')
        # Use existing thread or make new one
        thread_id = data.get('thread_id', str(uuid.uuid4()))

        config = {"configurable": {"thread_id": thread_id}}

        # Run the graph
        result = app.invoke(
            {"messages": [HumanMessage(content=message)]}, config=config)

        # Check if the AI hit an interrupt
        if '__interrupt__' in result:
            interrupt_data = result['__interrupt__'][0].value
            return Response({
                "status": "pending_approval",
                "thread_id": thread_id,
                "payload": interrupt_data
            })

        # If no interrupt, it's a normal text response
        last_msg = result['messages'][-1]
        final_message = getattr(last_msg, 'content', '')
        if isinstance(final_message, list):
            final_message = "".join(
                p.get('text', '') if isinstance(p, dict) else str(p) for p in final_message
            )
        if not final_message:
            final_message = "I have processed your request."

        return Response({
            "status": "success",
            "thread_id": thread_id,
            "message": final_message
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def approve_view(request):
    try:
        data = json.loads(request.body)
        thread_id = data.get('thread_id')
        decision = data.get('decision')  # 'approved' or 'rejected'

        config = {"configurable": {"thread_id": thread_id}}

        # Resume the graph with the user's decision
        result = app.invoke(Command(resume=decision), config=config)

        last_msg = result['messages'][-1]
        final_message = getattr(last_msg, 'content', '')
        if isinstance(final_message, list):
            final_message = "".join(
                p.get('text', '') if isinstance(p, dict) else str(p) for p in final_message
            )
        if not final_message:
            final_message = "Decision processed."

        return Response({
            "status": "success",
            "message": final_message
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
