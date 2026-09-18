"""
Service layer: LLM, embeddings, vector store, chat history.

Services are singletons — import the getters, not the raw objects.
"""

from app.services.llm import get_llm
from app.services.vector_store import get_vector_store, retrieve_documents
from app.services.chat_history import format_chat_history

__all__ = [
    "get_llm",
    "get_vector_store",
    "retrieve_documents",
    "format_chat_history",
]