"""Groq LLM singleton."""

from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(
        model=s.llm_model,
        temperature=s.temperature,
        api_key=s.groq_api_key,
    )