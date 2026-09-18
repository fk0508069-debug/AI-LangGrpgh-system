# app/nodes/retrieve.py
import logging
from app.state import GraphState
from app.services.vector_store import retrieve_documents

logger = logging.getLogger(__name__)

# Try each alias until we get hits. `None` = no filter.
CATEGORY_ALIASES = {
    "policy": ["policy", "policies"],
    "hr": ["hr", "HR", "human_resources"],
    "product": ["product", "products"],
}


def _category_candidates(query_type: str):
    if query_type in CATEGORY_ALIASES:
        # Try each alias, then fall back to no filter
        return CATEGORY_ALIASES[query_type] + [None]
    return [None]


def retrieve_node(state: GraphState) -> dict:
    query = state["question"]
    original_query = state.get("original_question", query)
    query_type = state.get("query_type", "unknown")

    candidates = _category_candidates(query_type)

    documents = []
    for category in candidates:
        documents = retrieve_documents(query=query, category=category)
        if documents:
            logger.info(f"Retrieved {len(documents)} docs with category={category!r}")
            break

    # If the rewritten query missed, try the original question
    if not documents and original_query != query:
        logger.warning(f"No results for '{query}'. Retrying original: '{original_query}'")
        for category in candidates:
            documents = retrieve_documents(query=original_query, category=category)
            if documents:
                break

    if not documents:
        logger.warning("No documents found for the query.")
        return {"documents": [], "context": ""}

    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        category = doc.metadata.get("category", "Unknown")
        context_parts.append(
            f"--- Document {i + 1} (Source: {source}, Page: {page}, Category: {category}) ---\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)
    return {"documents": documents, "context": context}