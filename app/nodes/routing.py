"""Routing nodes: load session, guard, detect intent, greeting, garbage."""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.services.session_store import session_store
from app.state import ConversationState, GraphState, Intent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phrase tables (kept identical to the original rag.py)
# ---------------------------------------------------------------------------
ORDER_INTENT_PHRASES = (
    "track this number",
    "track my order",
    "track order",
    "track my package",
    "track package",
    "trace my order",
    "tracking number",
    "tracking no",
    "tracking #",
    "order status",
    "where is my order",
    "where's my order",
    "where is my package",
    "where's my package",
    "track shipment",
    "delivery status",
    "when will my order arrive",
    "when will my package arrive",
    "when will it arrive",
    "when will it come",
    "when should it arrive",
    "when should i expect",
    "expected delivery",
    "delivery date",
)

ORDER_INTENT_PATTERNS = (
    r"\btrack(?:ing)?\b.*\b(?:number|no|#|order|package|shipment)\b",
    r"\b(?:order|package|shipment)\b.*\b(?:track|tracking|status|where)\b",
)

PRODUCT_INTENT_PHRASES = (
    # direct shopping verbs
    "recommend", "recommend me", "recommend some",
    "suggest", "suggest me", "suggest some",
    "show me", "find me", "find a", "find some", "find any",
    "looking for", "looking to buy",
    "i want", "i need", "i'd like", "i would like", "i wanna",
    "products", "product", "buy", "shopping", "shop",
    "do you have", "have any", "available", "find me some",
    # common product nouns so a bare noun works too
    "watch", "watches",
    "phone", "phones", "mobile", "smartphone",
    "laptop", "laptops", "computer", "pc",
    "shirt", "shirts", "tshirt", "tshirts", "hoodie",
    "shoes", "shoe", "sneakers", "sneaker", "boots",
    "headphone", "headphones", "earbuds", "earphone",
    "camera", "tablet", "tablets", "tv", "monitor", "keyboard", "mouse",
    "bag", "bags", "backpack", "wallet", "belt",
    "perfume", "fragrance",
)

TRACKING_FOLLOWUP_PHRASES = (
    "what is my tracking number",
    "what's my tracking number",
    "where is my tracking number",
    "show me my tracking number",
    "tell me my tracking number",
    "tracking number",
    "tracking no",
    "tracking #",
)

ORDER_STATUS_PHRASES = (
    "where is it",
    "where is my order",
    "where's my order",
    "where is my package",
    "where's my package",
    "order status",
    "delivery status",
    "status of my order",
)

ORDER_DELIVERY_PHRASES = (
    "when will it arrive",
    "when will it come",
    "when should it arrive",
    "when should i expect",
    "expected delivery",
    "delivery date",
    "when will my order arrive",
    "when will my package arrive",
)

CANCEL_INTENT_PATTERNS = (
    r"\b(?:cancel|cancell?ing|stop)\b.*\b(?:order|product|item|purchase|it)\b",
    r"\b(?:order|product|item|purchase)\b.*\b(?:cancel|cancell?ing|stop)\b",
    r"\b(?:want|need|would like)\s+to\s+cancel\b",
)

TRACKING_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9-]{6,30}(?![A-Za-z0-9]))"
    r"(?=[A-Za-z0-9-]*\d)([A-Za-z0-9-]{6,30})(?![A-Za-z0-9])"
)

VAGUE_QUERY_PATTERNS = [
    r"^(find|search|show|get|give|recommend|suggest|help)\s+(me\s+)?(something|anything|stuff|things?|products?|items?)$",
    r"^(i\s+)?(want|need|looking\s+for|am\s+looking\s+for)\s+(something|anything|stuff|things?|products?|items?)$",
    r"^(find|search|show|get|give)\s+(me\s+)?(some|any)\s+(products?|items?|stuff|things?)$",
    r"^(what|which)\s+(products?|items?)\s+(do\s+you\s+have|are\s+available)$",
    r"^(show|list)\s+(me\s+)?(all\s+)?(products?|items?)$",
    r"^(browse|explore)$",
    r"^(help\s+me\s+shop|help\s+me\s+find\s+something)$",
    r"^(i\s+need\s+help)$",
    r"^(i\s+want\s+to\s+buy\s+something)$",
]

VAGUE_KEYWORDS = {
    "something",
    "anything",
    "stuff",
    "things",
    "product",
    "products",
    "item",
    "items",
    "recommend",
    "suggest",
}

VAGUE_STOPWORDS = {
    "find",
    "search",
    "show",
    "me",
    "a",
    "an",
    "the",
    "some",
    "any",
    "for",
    "i",
    "want",
    "need",
    "looking",
    "help",
    "please",
    "can",
    "you",
    "get",
    "give",
    "buy",
}

IRRELEVANT_REPLY_PATTERNS = [
    r"^(yes|no|ok|okay|fine|sure|hmm+|umm+|idk|dunno|maybe|whatever|nothing|none)$",
    r"^(lol|haha+|hehe+|xd+)$",
    r"^[?.!,]+$",
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _contains_phrase(text: str, phrases: tuple) -> bool:
    return any(p in text for p in phrases)


def extract_tracking_number(message: str) -> Optional[str]:
    if not message:
        return None
    m = TRACKING_NUMBER_PATTERN.search(message)
    return m.group(1) if m else None


def is_order_tracking_request(message: str) -> bool:
    text = message.lower().strip()
    return _contains_phrase(text, ORDER_INTENT_PHRASES) or any(
        re.search(pattern, text) for pattern in ORDER_INTENT_PATTERNS
    )


def is_product_request(message: str) -> bool:
    return _contains_phrase(message.lower().strip(), PRODUCT_INTENT_PHRASES)


def is_cancel_request(message: str) -> bool:
    text = re.sub(r"\s+", " ", message.lower().strip())
    return any(re.search(p, text) for p in CANCEL_INTENT_PATTERNS)


def is_vague_query(question: str) -> bool:
    if not question:
        return True
    text = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", question.strip().lower())).strip()
    for p in VAGUE_QUERY_PATTERNS:
        if re.fullmatch(p, text):
            return True
    words = text.split()
    if len(words) <= 4:
        non_vague = [
            w for w in words if w not in VAGUE_KEYWORDS and w not in VAGUE_STOPWORDS
        ]
        if not non_vague:
            return True
    return False


def is_garbage_input(question: str) -> bool:
    q = question.strip()
    if not q or len(q) < 2:
        return True
    if not re.search(r"[A-Za-z0-9]", q):
        return True
    if not re.search(r"[A-Za-z]{2,}", q, re.IGNORECASE) and not re.search(r"\d", q):
        return True
    symbol_count = len(re.findall(r"[^A-Za-z0-9\s]", q))
    return len(q) > 0 and (symbol_count / len(q)) > 0.70


def is_irrelevant_reply(question: str) -> bool:
    text = re.sub(r"[^\w\s?!]", "", question.strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return True
    return any(re.fullmatch(p, text) for p in IRRELEVANT_REPLY_PATTERNS)


def detect_greeting(question: str) -> Optional[str]:
    if not question:
        return None
    text = re.sub(r"[^\w\s]", "", question.strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    thanks = [r"^(thanks?|thank\s+you|thank\s+u|thx|shukriya|shukriyaa)$"]
    bye = [r"^(bye|goodbye|good\s+bye|see\s+you|see\s+ya|take\s+care)$"]
    howru = [
        r"^(how\s+are\s+you|how\s+are\s+u|how\s+r\s+u|how\s+do\s+you\s+do)$",
        r"^(how\s+is\s+it\s+going|hows?\s+it\s+going)$",
        r"^(how\s+are\s+things)$",
    ]
    who = [
        r"^(who\s+are\s+you|what\s+are\s+you|whats?\s+your\s+name|what\s+is\s+your\s+name)$"
    ]
    greet = [
        r"^(hi|hii+|hey+|helo+|hello+|yo|hola|salam|assalamualaikum|asalamualaikum|asalam\s+o\s+alaikum|assalam\s+o\s+alaikum)$",
        r"^(good\s+morning|good\s+afternoon|good\s+evening|good\s+night)$",
        r"^(whats?\s+up|what\s+is\s+up|sup|wassup|whats\s+good)$",
        r"^(nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you)$",
    ]

    def matches(pats):
        return any(re.fullmatch(p, text, re.IGNORECASE) for p in pats)

    if matches(thanks):
        return "You're very welcome! 😊 If you need help with anything else, just ask."
    if matches(bye):
        return "Goodbye! 👋 Come back anytime. Happy shopping! 🛍️"
    if matches(howru):
        return "I'm great, thanks for asking! 😊 What can I help you find today?"
    if matches(who):
        return (
            "I'm your e-commerce AI assistant 🤖 I can help you find "
            "products and track orders. What would you like to do?"
        )
    if matches(greet):
        return (
            "Hello! 👋 Welcome to our store. How can I help you today? 😊 "
            "I can find products or track an order for you."
        )
    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def load_session_node(state: GraphState) -> dict:
    session_id = state["session_id"]
    session = session_store.get(session_id)
    logger.info("load_session | %s | state=%s", session_id, session.state.value)
    return {"session": session}


def guard_node(state: GraphState) -> dict:
    q = (state.get("original_question") or "").strip()
    if not q:
        return {"intent": Intent.EMPTY.value, "answer": "Please enter a question."}
    return {}


def detect_intent_node(state: GraphState) -> dict:
    session = state["session"]
    question = (state.get("original_question") or "").strip()

    # 1. Cancel confirmation takes priority
    if session.state == ConversationState.AWAITING_CANCEL_CONFIRMATION:
        return {"intent": Intent.CANCEL_CONFIRMATION.value}

    # 2. Pending clarification
    if session.state == ConversationState.AWAITING_CLARIFICATION:
        return {"intent": Intent.PRODUCT.value}

    # 3. Garbage
    if is_garbage_input(question):
        return {"intent": Intent.GARBAGE.value}

    # 4. Greeting
    greeting = detect_greeting(question)
    if greeting:
        return {"intent": Intent.GREETING.value, "greeting_response": greeting}

    # 5. Cancel
    if is_cancel_request(question):
        return {"intent": Intent.ORDER_CANCEL.value}

    # 6. Tracking number present
    tnum = extract_tracking_number(question)
    if tnum:
        return {"intent": Intent.ORDER_TRACKING.value, "tracking_number": tnum}

    # 7. Followup on existing order in session
    if session.tracking_number:
        text = question.lower().strip()
        if (
            _contains_phrase(text, TRACKING_FOLLOWUP_PHRASES)
            or _contains_phrase(text, ORDER_STATUS_PHRASES)
            or _contains_phrase(text, ORDER_DELIVERY_PHRASES)
        ):
            return {"intent": Intent.ORDER_FOLLOWUP.value}

    # 8. Asked to track but no number yet
    if is_order_tracking_request(question):
        return {"intent": Intent.ORDER_TRACKING_PROMPT.value}

    # 9. Product / vague
    if is_product_request(question) or is_vague_query(question):
        return {"intent": Intent.PRODUCT.value}

    # 10. Default
    return {"intent": Intent.RAG.value}


def greeting_node(state: GraphState) -> dict:
    return {"answer": state.get("greeting_response") or "Hello! 😊"}


def garbage_node(state: GraphState) -> dict:
    return {
        "answer": "I'm sorry, I didn't catch that. 😊 Could you rephrase your request?"
    }


__all__ = [
    "load_session_node",
    "guard_node",
    "detect_intent_node",
    "greeting_node",
    "garbage_node",
    "extract_tracking_number",
    "is_order_tracking_request",
    "is_product_request",
    "is_cancel_request",
    "is_vague_query",
    "is_garbage_input",
    "is_irrelevant_reply",
    "detect_greeting",
    # phrase tables (imported by orders.py)
    "TRACKING_FOLLOWUP_PHRASES",
    "ORDER_STATUS_PHRASES",
    "ORDER_DELIVERY_PHRASES",
    "_contains_phrase",
]
