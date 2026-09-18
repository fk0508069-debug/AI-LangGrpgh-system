"""Persist a completed turn into session history."""

from __future__ import annotations

import logging

from app.services.session_store import session_store
from app.state import GraphState

logger = logging.getLogger(__name__)


def save_turn_node(state: GraphState) -> dict:
    session_id = state["session_id"]
    question = state.get("original_question") or ""
    answer = state.get("answer") or ""

    if not answer:
        logger.warning("save_turn: empty answer, skipping persist.")
        return {}

    session_store.save_turn(session_id, question, answer)
    return {}


__all__ = ["save_turn_node"]