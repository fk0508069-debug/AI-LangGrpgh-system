"""LangGraph wiring — the whole state machine lives here."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.nodes import (
    cancel_confirmation_node,
    cancel_request_node,
    clarify_node,
    detect_intent_node,
    garbage_node,
    generate_product_answer_node,
    greeting_node,
    guard_node,
    load_session_node,
    merge_results_node,
    order_followup_node,
    order_tracking_node,
    order_tracking_prompt_node,
    rag_generate_node,
    rag_retrieve_node,
    save_turn_node,
    search_keyword_node,
    search_semantic_node,
    search_structured_node,
    understand_query_node,
)
from app.state import GraphState, Intent

logger = logging.getLogger(__name__)


def _route_after_intent(state: GraphState) -> str:
    """Map the detected intent to the next node name."""
    intent = state.get("intent") or Intent.RAG.value
    mapping = {
        Intent.EMPTY.value: "save_turn",
        Intent.GARBAGE.value: "garbage",
        Intent.GREETING.value: "greeting",
        Intent.ORDER_TRACKING.value: "order_tracking",
        Intent.ORDER_TRACKING_PROMPT.value: "order_tracking_prompt",
        Intent.ORDER_FOLLOWUP.value: "order_followup",
        Intent.ORDER_CANCEL.value: "cancel_request",
        Intent.CANCEL_CONFIRMATION.value: "cancel_confirmation",
        Intent.PRODUCT.value: "understand_query",
        Intent.RAG.value: "rag_retrieve",
    }
    return mapping.get(intent, "rag_retrieve")


def _route_after_understand(state: GraphState) -> str:
    """After query-understanding: clarify, or run the product search."""
    parsed = state.get("parsed_query") or {}
    if not parsed.get("valid", False):
        return "rag_retrieve"
    if parsed.get("needs_clarification"):
        return "clarify"
    return "search_structured"


def build_graph():
    workflow = StateGraph(GraphState)

    # --- Nodes ---
    workflow.add_node("load_session", load_session_node)
    workflow.add_node("guard", guard_node)
    workflow.add_node("detect_intent", detect_intent_node)

    workflow.add_node("greeting", greeting_node)
    workflow.add_node("garbage", garbage_node)

    workflow.add_node("order_tracking", order_tracking_node)
    workflow.add_node("order_tracking_prompt", order_tracking_prompt_node)
    workflow.add_node("order_followup", order_followup_node)
    workflow.add_node("cancel_request", cancel_request_node)
    workflow.add_node("cancel_confirmation", cancel_confirmation_node)

    workflow.add_node("understand_query", understand_query_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("search_structured", search_structured_node)
    workflow.add_node("search_keyword", search_keyword_node)
    workflow.add_node("search_semantic", search_semantic_node)
    workflow.add_node("merge_results", merge_results_node)
    workflow.add_node("generate_product_answer", generate_product_answer_node)

    workflow.add_node("rag_retrieve", rag_retrieve_node)
    workflow.add_node("rag_generate", rag_generate_node)

    workflow.add_node("save_turn", save_turn_node)

    # --- Edges ---
    workflow.add_edge(START, "load_session")
    workflow.add_edge("load_session", "guard")
    workflow.add_edge("guard", "detect_intent")

    workflow.add_conditional_edges(
        "detect_intent",
        _route_after_intent,
        {
            "save_turn": "save_turn",
            "garbage": "garbage",
            "greeting": "greeting",
            "order_tracking": "order_tracking",
            "order_tracking_prompt": "order_tracking_prompt",
            "order_followup": "order_followup",
            "cancel_request": "cancel_request",
            "cancel_confirmation": "cancel_confirmation",
            "understand_query": "understand_query",
            "rag_retrieve": "rag_retrieve",
        },
    )

    # Every terminal-from-intent node funnels to save_turn
    for node in [
        "garbage",
        "greeting",
        "order_tracking",
        "order_tracking_prompt",
        "order_followup",
        "cancel_request",
        "cancel_confirmation",
    ]:
        workflow.add_edge(node, "save_turn")

    # Product pipeline
    workflow.add_conditional_edges(
        "understand_query",
        _route_after_understand,
        {
            "clarify": "clarify",
            "search_structured": "search_structured",
            "rag_retrieve": "rag_retrieve",
        },
    )
    workflow.add_edge("clarify", "save_turn")
    workflow.add_edge("search_structured", "search_keyword")
    workflow.add_edge("search_keyword", "search_semantic")
    workflow.add_edge("search_semantic", "merge_results")
    workflow.add_edge("merge_results", "generate_product_answer")
    workflow.add_edge("generate_product_answer", "save_turn")

    # RAG fallback
    workflow.add_edge("rag_retrieve", "rag_generate")
    workflow.add_edge("rag_generate", "save_turn")

    # Terminal
    workflow.add_edge("save_turn", END)

    logger.info("LangGraph compiled successfully.")
    return workflow.compile()