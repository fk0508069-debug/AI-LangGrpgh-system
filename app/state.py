"""GraphState + persistent SessionData."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class ConversationState(str, Enum):
    IDLE = "idle"
    AWAITING_TRACKING = "awaiting_tracking"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    ORDER_FOLLOWUP = "order_followup"
    AWAITING_CANCEL_CONFIRMATION = "awaiting_cancel_confirmation"


class Intent(str, Enum):
    EMPTY = "empty"
    GARBAGE = "garbage"
    GREETING = "greeting"
    ORDER_TRACKING = "order_tracking"
    ORDER_TRACKING_PROMPT = "order_tracking_prompt"
    ORDER_FOLLOWUP = "order_followup"
    ORDER_CANCEL = "order_cancel"
    CANCEL_CONFIRMATION = "cancel_confirmation"
    PRODUCT = "product"
    RAG = "rag"


@dataclass
class SessionData:
    """Persistent state, lives in session_store, keyed by session_id."""

    history: List[BaseMessage] = field(default_factory=list)
    state: ConversationState = ConversationState.IDLE

    tracking_number: Optional[str] = None
    order_data: Optional[dict] = None

    pending_clarification: Optional[str] = None
    last_parsed: Optional[dict] = None
    clarification_attempts: int = 0
    last_products: List[dict] = field(default_factory=list)

    _vocabulary_cache: Optional[Dict[str, List[str]]] = None
    _vocabulary_cache_time: Optional[datetime] = None


class GraphState(TypedDict, total=False):
    """Passed between nodes. Everything optional because branches differ."""

    # routing
    session_id: str
    session: SessionData
    original_question: str
    question: str
    intent: str
    error: Optional[str]

    # memory
    chat_history: Annotated[List[BaseMessage], operator.add]

    # greeting
    greeting_response: Optional[str]

    # order
    tracking_number: Optional[str]
    order_data: Optional[dict]
    cancel_confirmed: bool

    # product pipeline
    query_type: str
    parsed_query: dict
    needs_clarification: bool
    clarification_question: Optional[str]

    # search
    structured_products: List[dict]
    keyword_products: List[dict]
    products: List[dict]
    documents: List[Document]
    context: str

    # generation
    answer: str
    validation_result: bool