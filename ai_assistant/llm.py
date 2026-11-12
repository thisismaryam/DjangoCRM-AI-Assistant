import os

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from ai_assistant.tools import tools

load_dotenv()


def build_fallback_llm(primary: Runnable, fallbacks: list[Runnable]) -> Runnable:
    """Chains the primary model to a list of fallback models in order."""
    return primary.with_fallbacks(fallbacks)


# 1. Primary Gemini Model
gemini_primary = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=512,
    retries=0,
).bind_tools(tools, parallel_tool_calls=False)

# 2. Fallback Gemini Model
gemini_fallback = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash"),
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=512,
    retries=0,
).bind_tools(tools, parallel_tool_calls=False)

# 3. Fallback Groq Model
groq_fallback = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=512,
).bind_tools(tools, parallel_tool_calls=False)


# 4. Chain them: Primary -> Fallback Gemini -> Fallback Groq
llm = build_fallback_llm(
    primary=gemini_primary,
    fallbacks=[gemini_fallback, groq_fallback]
)
