# app/services/llm.py
import logging
from langchain_groq import ChatGroq
from app.config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY

logger = logging.getLogger(__name__)

_llm = None


def get_llm() -> ChatGroq:
    """Singleton pattern to initialize the Groq LLM once."""
    global _llm
    if _llm is None:
        logger.info(f"Initializing Groq LLM: {LLM_MODEL}")
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
            max_tokens=1024,
        )
    return _llm
