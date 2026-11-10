import os

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from ai_assistant.tools import tools


load_dotenv()


def build_fallback_llm(primary: Runnable, fallback: Runnable) -> Runnable:
    return primary.with_fallbacks([fallback])


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=512,
    retries=0,
)
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=512,
)

llm = build_fallback_llm(
    gemini_llm.bind_tools(tools, parallel_tool_calls=False),
    groq_llm.bind_tools(tools, parallel_tool_calls=False),
)
