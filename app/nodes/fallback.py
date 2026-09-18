# app/nodes/fallback.py
import logging
from app.state import GraphState

logger = logging.getLogger(__name__)


def fallback_node(state: GraphState) -> dict:
    """
    Fallback when validation fails. We keep the LLM's original answer
    but attach a soft disclaimer, so the user still gets real value.
    """
    logger.warning("Validation failed. Routing to fallback.")

    original = (state.get("answer") or "").strip()

    if original:
        answer = (
            f"{original}\n\n"
            "_(Note: this answer could not be fully verified against the company "
            "documents. Please double-check with HR if it is critical.)_"
        )
    else:
        answer = (
            "I couldn't find a reliable answer in the company documents. "
            "Please rephrase your question or contact HR directly."
        )

    return {"answer": answer}