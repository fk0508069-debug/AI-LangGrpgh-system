"""RAG fallback: retrieve context, generate answer."""

from __future__ import annotations

import logging

from app.prompts import RAG_PROMPT
from app.services.helpers import safe_string
from app.services.llm import get_llm
from app.services.vector_store import get_retriever
from app.state import GraphState

logger = logging.getLogger(__name__)

# Groq compound-mini caps the request around 4-6K chars.
# Keep the whole context small to be safe.
MAX_DOC_CHARS = 500
MAX_CONTEXT_CHARS = 1500


def rag_retrieve_node(state: GraphState) -> dict:
    question = state.get("original_question") or ""
    try:
        retriever = get_retriever()
        docs = retriever.invoke(question)
    except Exception as error:
        logger.error("Retrieval failed: %s", error)
        docs = []

    # Keep only 1-3 docs and truncate each
    trimmed = []
    total = 0
    for d in docs:
        if not d.page_content:
            continue
        chunk = d.page_content[:MAX_DOC_CHARS]
        if total + len(chunk) > MAX_CONTEXT_CHARS:
            break
        trimmed.append(d)
        total += len(chunk)

    context = "\n\n".join(
        f"Source: {(d.metadata or {}).get('source', 'knowledge base')}\n"
        f"{d.page_content[:MAX_DOC_CHARS]}"
        for d in trimmed
    ) or "No relevant information was found in the knowledge base."

    logger.info("RAG context: %d docs, %d chars", len(trimmed), len(context))
    return {"documents": trimmed, "context": context}


def rag_generate_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""
    context = state.get("context") or ""

    # Hard cap as belt-and-suspenders
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    llm = get_llm()

    base_messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    # Do NOT include chat history for RAG — it makes the request huge
    # and the previous answers (including failures) confuse the model.
    invoke_messages = base_messages

    logger.info(
        "RAG request | context=%d chars | messages=%d",
        len(context),
        len(invoke_messages),
    )

    try:
        response = llm.invoke(invoke_messages)
        answer = safe_string(
            response.content, "I don't know based on the available information."
        ).strip()
    except Exception as error:
        # Log the FULL error so we can actually see what Groq says
        logger.error("RAG generation failed: %s: %s", type(error).__name__, error)
        # Short honest fallback — do NOT dump raw context into history
        if context and not context.startswith("No relevant information"):
            answer = (
                "I found policy text that may be relevant, but I couldn't "
                "generate a summary right now. Please try again in a moment."
            )
        else:
            answer = (
                "I couldn't find that in the knowledge base. "
                "Try rephrasing, or ask about a specific policy."
            )

    return {"answer": answer}


__all__ = ["rag_retrieve_node", "rag_generate_node"]