from ai_assistant.agent import app
from langgraph.types import Command
from langchain_core.messages import HumanMessage
import os
import django
import uuid
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webcrm.settings')
django.setup()


def test_interrupt():
    print("--- Test: Human-in-the-loop ---")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    inputs = {"messages": [HumanMessage(
        content="Use the prepare_lead tool to create a lead for John Doe, email john@acme.com, company Acme Corp")]}

    print("\n[1. Running Graph...]")
    # We use invoke instead of stream for the first run to see the full result
    try:
        result = app.invoke(inputs, config=config)
        print("\nGraph finished. Result:")
        print(result)
    except Exception as e:
        print(
            f"\nGraph paused with an exception (this is expected for interrupts): {type(e).__name__}")
        print(e)

    # Let's check the state manually
    state = app.get_state(config)
    print("\n[2. Checking Graph State]")
    print(f"Next node to execute: {state.next}")
    print(f"Full State Values: {state.values}")


if __name__ == "__main__":
    test_interrupt()
