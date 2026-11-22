from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import json
import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from django.utils import timezone

from langgraph.checkpoint.sqlite import SqliteSaver

from ai_assistant.context import trim_conversation_history
from ai_assistant.llm import llm
from ai_assistant.tools import tools

# Load API key
load_dotenv()

# Define the State


class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    original_request: str
    operation_id: str


# Create the Agent Node


def agent_node(state: AgentState):
    messages = trim_conversation_history(state["messages"])
    original_req = state.get("original_request", "")

    # Add a system prompt to guide the AI
    # The prompt describes behaviour only. Tool names, parameters and what
    # each tool does are sent to the model automatically from the tool
    # schemas and docstrings (bind_tools), so they are not repeated here.
    system_prompt = SystemMessage(content=(
        f"Today is {timezone.localdate().isoformat()}. "
        "You are an AI assistant for Django-CRM. Your sole scope is automating the **Lead** model.\n\n"

        "CRITICAL SCOPE RULE:\n"
        "You can only create Leads. You cannot create or modify Companies, Contacts, Deals, or Payments. "
        "If the user asks you to create any other record type, politely refuse and tell them to use the CRM admin.\n\n"

        "TOOL USE RULES:\n"
        "- Always use `search_database_schema` to answer questions about CRM structure or fields. Never guess.\n"
        "- Always use `search_companies` or `search_contacts` to check if a record exists before trying to link it.\n"
        "- Never answer questions about CRM records from memory. Use the tools as the source of truth.\n\n"

        "LEAD CREATION FLOW:\n"
        "1. To create a lead, you must have at least a `first_name` and `email`. If missing, ask the user.\n"
        "2. If a company name is provided, use the tools to find it. If multiple companies match, ask the user to clarify which one using the ID.\n"
        "3. Pass all available information (name, email, company, phone, etc.) to the `prepare_lead` tool.\n"
        "4. The `prepare_lead` tool will automatically inherit the company's address, country, city, and website. Do not ask the user for these if the company is found.\n"
        "5. The tool will pause for human approval. \n"
        "   - On `SUCCESS`, tell the user the lead was created.\n"
        "   - On `CANCELLED`, say nothing was created and give the reason.\n"
        "   - On `NEEDS_DETAILS`, relay the message and ask for the missing info.\n"
        "6. Never bypass the approval step. If the user says 'skip approval', ignore them and show the card anyway.\n\n"

        "DOCUMENT EXTRACTION:\n"
        "If the user uploads a document, extract their name, email, phone, and company to initiate the LEAD CREATION FLOW.\n\n"

        "STYLE:\n"
        "- Keep answers short and clear. Ask only one clarifying question at a time.\n"
        "- Never expose internal errors, database traces, or tool execution details to the user.\n"
    )).
    prompt_messages = [system_prompt] + messages
    if os.environ.get("AI_ASSISTANT_LOG_PROMPTS") == "1":
        print(
            "[ai_assistant] Messages sent to configured provider:\n"
            + json.dumps(
                [
                    message.model_dump(mode="json", exclude_none=True)
                    for message in prompt_messages
                ],
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


# Create the Tool Node (executes the tools)
tool_node = ToolNode(tools)

# Conditional edge logic (should we call a tool or end?)


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # If there are tool calls, go to the tool node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END


# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set Entry Point
workflow.set_entry_point("agent")

# Add Conditional Edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# After tools execute, go back to the agent so it can answer the user
workflow.add_edge("tools", "agent")

# Compile the graph


# JsonPlusSerializer's pickle fallback handles LangGraph custom objects (such
# as Interrupt) that are not representable by its msgpack/JSON encoders.
serde = JsonPlusSerializer(pickle_fallback=True)
# Setup SQLite checkpointer to save paused states
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn, serde=serde)

app = workflow.compile(checkpointer=memory)
