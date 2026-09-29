"""Persist a completed turn into session history."""

from __future__ import annotations

import logging

from app.services.session_store import session_store
from app.state import GraphState

logger = logging.getLogger(__name__)


def save_turn_node(state: GraphState) -> dict:
    """Save the completed turn (question + answer) into session history.

    The store's signature is ``save_turn(session_id, question, answer)`` —
    no metadata. Topic separation is already handled upstream by
    ``session.pending_clarification`` in the product pipeline, so we don't
    duplicate that work here.
    """
    session_id = state["session_id"]
    question = (state.get("original_question") or "").strip()
    answer = (state.get("answer") or "").strip()

    if not answer:
        logger.warning("save_turn: empty answer, skipping persist.")
        return {}

    try:
        session_store.save_turn(session_id, question, answer)
    except Exception:
        # Never let a persistence hiccup fail the graph run.
        logger.exception("save_turn: persist failed for session=%s", session_id)

    return {}


__all__ = ["save_turn_node"]