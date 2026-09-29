"""RAG fallback: retrieve context, generate answer."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.prompts import RAG_PROMPT
from app.services.helpers import safe_string
from app.services.llm import get_llm
from app.services.vector_store import get_retriever
from app.state import GraphState

logger = logging.getLogger(__name__)

# Groq compound-mini caps the request around 4-6K chars.
MAX_DOC_CHARS = 500
MAX_CONTEXT_CHARS = 1500
MAX_QUESTION_CHARS = 500  # defend against oversized user input

_EMPTY_CONTEXT = "No relevant information was found in the knowledge base."

_llm_client = None


def _llm():
    """Lazy singleton — avoids re-instantiating the client on every RAG hit."""
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm()
    return _llm_client


def _source_label(metadata: Optional[Dict[str, Any]]) -> str:
    """Return a short, safe source label (filename only, no full path)."""
    raw = (metadata or {}).get("source") or "knowledge base"
    # Strip both POSIX and Windows path separators.
    return str(raw).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------
def rag_retrieve_node(state: GraphState) -> dict:
    question = state.get("original_question") or ""

    try:
        docs = get_retriever().invoke(question) or []
    except Exception as error:
        logger.error(
            "RAG retrieval failed: %s: %s", type(error).__name__, error
        )
        docs = []

    if not docs:
        return {"documents": [], "context": _EMPTY_CONTEXT}

    # Single pass: trim, budget, and format in one go.
    parts: List[str] = []
    kept_docs: List[Any] = []
    used = 0

    for d in docs:
        content = (d.page_content or "").strip()
        if not content:
            continue

        header = f"Source: {_source_label(d.metadata)}\n"
        # "-2" accounts for the "\n\n" separator this entry will add.
        remaining = MAX_CONTEXT_CHARS - used - len(header) - 2
        if remaining <= 0:
            break

        # Truncate to fit remaining budget — do not skip a relevant doc just
        # because it's slightly too large.
        chunk = content[: min(MAX_DOC_CHARS, remaining)]
        entry = header + chunk

        parts.append(entry)
        kept_docs.append(d)
        used += len(entry) + 2

    if not parts:
        return {"documents": [], "context": _EMPTY_CONTEXT}

    context = "\n\n".join(parts)
    logger.debug(
        "RAG context built: %d docs, %d chars", len(kept_docs), len(context)
    )
    return {"documents": kept_docs, "context": context}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def rag_generate_node(state: GraphState) -> dict:
    question = (state.get("original_question") or "").strip()
    context = (state.get("context") or "").strip()

    # Fast path: no context → no LLM round-trip.
    if not context or context == _EMPTY_CONTEXT:
        return {
            "answer": (
                "I couldn't find that in the knowledge base. "
                "Try rephrasing, or ask about a specific policy."
            )
        }

    # Defence in depth — context is already capped upstream.
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    if len(question) > MAX_QUESTION_CHARS:
        logger.warning(
            "Trimming oversized question (%d chars)", len(question)
        )
        question = question[:MAX_QUESTION_CHARS]

    messages = RAG_PROMPT.format_messages(context=context, question=question)
    logger.debug(
        "RAG request | context=%d chars | question=%d chars",
        len(context), len(question),
    )

    try:
        response = _llm().invoke(messages)
        answer = safe_string(
            response.content,
            "I don't know based on the available information.",
        ).strip()
    except Exception as error:
        logger.error(
            "RAG generation failed: %s: %s", type(error).__name__, error
        )
        # Short honest fallback — never dump raw context into the answer.
        answer = (
            "I found policy text that may be relevant, but I couldn't "
            "generate a summary right now. Please try again in a moment."
        )

    return {"answer": answer}


__all__ = ["rag_retrieve_node", "rag_generate_node"]