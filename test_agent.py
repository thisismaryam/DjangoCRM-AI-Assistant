from ai_assistant.agent import app
from langchain_core.messages import HumanMessage
import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webcrm.settings')
django.setup()


def run_test():
    print("Starting Agent...")
    print("\n--- Test 1: Schema Search ---")

    inputs = {"messages": [HumanMessage(
        content="Where do I store sales deals?")]}

    # Run the graph
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            if "messages" in value:
                msg = value["messages"][-1]
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(
                        f"  [Tool Call]: {msg.tool_calls[0]['name']} with args {msg.tool_calls[0]['args']}")
                elif msg.content:
                    print(f"  [Answer]: {msg.content}")
    print("\n" + "="*40)


if __name__ == "__main__":
    run_test()
