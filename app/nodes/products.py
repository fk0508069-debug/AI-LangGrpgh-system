"""Product pipeline: understand → clarify → search → merge → generate."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.prompts import (
    CLARIFICATION_PROMPT,
    PRODUCT_ANSWER_PROMPT,
    QUERY_UNDERSTANDING_PROMPT,
)
from app.services.helpers import format_money, safe_float, safe_int, safe_string
from app.services.llm import get_llm
from app.services.mongodb import mongodb
from app.services.vector_store import get_vectorstore
from app.state import ConversationState, GraphState

logger = logging.getLogger(__name__)

MAX_SUGGESTION_ITEMS = 6
CATEGORY_CACHE_TTL = timedelta(minutes=30)

VOCAB_FIELDS = [
    "category", "subcategory", "subsubcategory", "brand",
    "color", "size", "material", "gender",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _history_as_dicts(session) -> List[Dict[str, str]]:
    from langchain_core.messages import AIMessage
    out: List[Dict[str, str]] = []
    for m in session.history:
        role = "assistant" if isinstance(m, AIMessage) else "user"
        out.append({"role": role, "content": str(m.content)})
    return out


def _fetch_vocabulary() -> Dict[str, List[str]]:
    vocab: Dict[str, List[str]] = {}
    for field in VOCAB_FIELDS:
        try:
            values = mongodb.distinct(mongodb.products, field)
            vocab[field] = sorted(
                {str(v).strip() for v in values if v and str(v).strip()}
            )
        except Exception:
            vocab[field] = []
    return vocab


def _get_vocabulary(session) -> Dict[str, List[str]]:
    now = datetime.now()
    if (
        session._vocabulary_cache is not None
        and session._vocabulary_cache_time is not None
        and now - session._vocabulary_cache_time < CATEGORY_CACHE_TTL
    ):
        return session._vocabulary_cache
    try:
        vocab = _fetch_vocabulary()
    except Exception:
        logger.exception("Vocab fetch failed.")
        vocab = {}
    session._vocabulary_cache = vocab
    session._vocabulary_cache_time = now
    return vocab


def _build_mongo_filter(parsed: Dict[str, Any]) -> Dict[str, Any]:
    mongo_filter: Dict[str, Any] = {}
    min_price = parsed.get("min_price")
    max_price = parsed.get("max_price")
    if min_price is not None or max_price is not None:
        pf = {}
        if min_price is not None:
            pf["$gte"] = safe_float(min_price)
        if max_price is not None:
            pf["$lte"] = safe_float(max_price)
        mongo_filter["price"] = pf
    if parsed.get("in_stock") is True:
        mongo_filter["stock"] = {"$gt": 0}
    for field in VOCAB_FIELDS:
        val = parsed.get(field)
        if val:
            mongo_filter[field] = {"$regex": re.escape(str(val)), "$options": "i"}
    return mongo_filter


def _get_product_url(product_id: Any) -> str:
    from app.config import get_settings
    try:
        pid = str(product_id).strip()
        if pid.startswith("ObjectId("):
            pid = pid.replace("ObjectId(", "").replace(")", "").strip("'\"")
        base = get_settings().product_base_url.rstrip("/")
        return f"{base}/{pid}"
    except Exception:
        return ""


def _build_product_context(products: List[Dict[str, Any]]) -> str:
    if not products:
        return "NO MATCHING PRODUCTS WERE FOUND."
    sections = []
    for index, p in enumerate(products, 1):
        pid = safe_string(p.get("_id"))
        link = _get_product_url(pid) if pid else "Not available"
        sections.append(
            f"""
PRODUCT {index}
PRODUCT ID: {pid}
PRODUCT LINK: {link}
NAME: {safe_string(p.get('name'))}
PRICE: Rs. {format_money(p.get('price'))}
STOCK: {safe_int(p.get('stock'))}
CATEGORY: {safe_string(p.get('category'))}
SUBCATEGORY: {safe_string(p.get('subcategory'))}
BRAND: {safe_string(p.get('brand'))}
COLOR: {safe_string(p.get('color'))}
SIZE: {safe_string(p.get('size'))}
MATERIAL: {safe_string(p.get('material'))}
GENDER: {safe_string(p.get('gender'))}
DESCRIPTION: {safe_string(p.get('description'))}
""".strip()
        )
    return "\n\n" + "\n\n---\n\n".join(sections)


def _format_products_fallback(products: list) -> str:
    if not products:
        return (
            "I couldn't find any products matching that right now. "
            "Try a different keyword or widen your budget. 😊"
        )
    lines = ["Here are some options for you 😊", ""]
    for index, product in enumerate(products[:MAX_SUGGESTION_ITEMS], 1):
        name = safe_string(product.get("name"), "Product")
        price = format_money(product.get("price"))
        category = safe_string(product.get("category"))
        color = safe_string(product.get("color"))
        stock = product.get("stock")
        pid = product.get("_id")
        url = _get_product_url(pid) if pid else ""

        lines.append(f"{index}. **{name}**")
        lines.append(f"   • Price: Rs. {price}")
        if stock is not None:
            lines.append(f"   • Stock: {stock}")
        if category or color:
            bits = [b for b in [category, color] if b]
            lines.append(f"   • {' | '.join(bits)}")
        if pid:
            lines.append(f"   • Product ID: {pid}")
        if url:
            lines.append(f"   • Link: {url}")
        lines.append("")
    lines.append("Want me to narrow this down by color, brand, or budget? 💙")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: understand query
# ---------------------------------------------------------------------------
def understand_query_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""

    # Was the user in the middle of a clarification?
    pending = session.pending_clarification
    if pending:
        merged = f"Previous request: {pending}\nUser's clarification: {question}"
    else:
        merged = question

    vocab = _get_vocabulary(session)
    llm = get_llm()

    # Build the system message directly from the prompt template
    vocab_json = json.dumps(vocab, ensure_ascii=False)
    system_msg = QUERY_UNDERSTANDING_PROMPT.format_messages(
        vocabulary=vocab_json,
        question=merged,
    )[0]  # the system message

    # Assemble: system + last 6 turns of history + current user turn
    messages = [system_msg]
    for turn in _history_as_dicts(session)[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": merged})

    try:
        response = llm.invoke(messages)
        content = safe_string(response.content).strip()
        content = re.sub(r"```json|```", "", content, flags=re.IGNORECASE).strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Non-dict response")
    except Exception as error:
        logger.error("Query understanding failed: %s", error)
        parsed = {
            "valid": False,
            "intent": "general",
            "keywords": [],
            "needs_clarification": False,
        }

    # Merge with prior parse if we were clarifying
    if session.last_parsed and pending:
        for key in VOCAB_FIELDS + ["min_price", "max_price", "sort", "in_stock"]:
            if parsed.get(key) in (None, [], ""):
                parsed[key] = session.last_parsed.get(key)
        prior_kw = session.last_parsed.get("keywords") or []
        new_kw = parsed.get("keywords") or []
        if isinstance(prior_kw, list) and isinstance(new_kw, list):
            parsed["keywords"] = list(dict.fromkeys([*prior_kw, *new_kw]))

    return {"parsed_query": parsed}

# ---------------------------------------------------------------------------
# Node: clarify
# ---------------------------------------------------------------------------
def clarify_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""
    parsed = state.get("parsed_query") or {}

    # Guard against garbage replies
    from app.nodes.routing import is_garbage_input, is_irrelevant_reply
    if session.pending_clarification and (is_garbage_input(question) or is_irrelevant_reply(question)):
        session.clarification_attempts += 1
        if session.clarification_attempts >= 2:
            session.clarification_attempts = 0
            return {
                "answer": (
                    "I'm not sure I understood that. 😊 Could you describe "
                    "what you're looking for — for example, a category, a product type, or your budget?"
                )
            }
        return {
            "answer": (
                "Sorry, I didn't understand that. Could you tell me a bit more "
                "about what you'd like to shop for?"
            )
        }

    # Prefer the LLM-suggested clarification question if present
    clarification = parsed.get("clarification_question")
    if not clarification:
        vocab = _get_vocabulary(session)
        history = _history_as_dicts(session)[-8:]
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history)
        llm = get_llm()
        chain = CLARIFICATION_PROMPT | llm
        try:
            resp = chain.invoke({
                "history": history_text,
                "question": question,
                "categories": vocab.get("category", [])[:12],
                "brands": vocab.get("brand", [])[:12],
            })
            clarification = safe_string(resp.content).strip()
        except Exception as error:
            logger.error("Clarification generation failed: %s", error)
            clarification = (
                "Could you share a bit more — for example, the product type "
                "or your budget? 😊"
            )

    session.pending_clarification = question
    session.last_parsed = parsed
    session.state = ConversationState.AWAITING_CLARIFICATION
    return {"answer": clarification, "needs_clarification": True}


# ---------------------------------------------------------------------------
# Nodes: search
# ---------------------------------------------------------------------------
def search_structured_node(state: GraphState) -> dict:
    parsed = state.get("parsed_query") or {}
    try:
        mongo_filter = _build_mongo_filter(parsed)
        cursor = mongodb.products.find(mongo_filter)
        sort_type = parsed.get("sort")
        if sort_type == "price_asc":
            cursor = cursor.sort("price", 1)
        elif sort_type == "price_desc":
            cursor = cursor.sort("price", -1)
        else:
            cursor = cursor.sort("createdAt", -1)
        results = list(cursor.limit(MAX_SUGGESTION_ITEMS))
    except Exception:
        logger.exception("Structured search failed.")
        results = []
    return {"structured_products": results}


def search_keyword_node(state: GraphState) -> dict:
    parsed = state.get("parsed_query") or {}
    keywords = parsed.get("keywords") or []
    product_name = parsed.get("product_name")
    terms = []
    if product_name:
        terms.append(str(product_name))
    if isinstance(keywords, list):
        terms.extend(keywords)
    terms = [str(t).strip() for t in terms if str(t).strip()]
    if not terms:
        return {"keyword_products": []}
    try:
        regex_conditions = []
        for term in terms:
            pattern = {"$regex": re.escape(term), "$options": "i"}
            regex_conditions.extend([
                {"name": pattern},
                {"description": pattern},
                {"category": pattern},
                {"subcategory": pattern},
                {"subsubcategory": pattern},
                {"brand": pattern},
                {"color": pattern},
            ])
        results = list(
            mongodb.products.find({"$or": regex_conditions}).limit(MAX_SUGGESTION_ITEMS)
        )
    except Exception:
        logger.exception("Keyword search failed.")
        results = []
    return {"keyword_products": results}


def search_semantic_node(state: GraphState) -> dict:
    question = state.get("original_question") or ""
    try:
        vs = get_vectorstore()
        results = vs.similarity_search(question, k=5)
    except Exception as error:
        logger.error("Semantic search failed: %s", error)
        results = []
    return {"documents": results}


# ---------------------------------------------------------------------------
# Node: merge
# ---------------------------------------------------------------------------
def merge_results_node(state: GraphState) -> dict:
    structured = state.get("structured_products") or []
    keyword = state.get("keyword_products") or []
    documents = state.get("documents") or []

    products: Dict[str, Dict[str, Any]] = {}
    for p in structured:
        products[str(p.get("_id"))] = p
    for p in keyword:
        pid = str(p.get("_id"))
        if pid not in products:
            products[pid] = p
    for doc in documents:
        mongo_id = (doc.metadata or {}).get("mongo_id")
        if mongo_id and mongo_id not in products:
            prod = mongodb.get_product_by_id(mongo_id)
            if prod:
                products[mongo_id] = prod

    merged = list(products.values())[:MAX_SUGGESTION_ITEMS]
    return {"products": merged}


# ---------------------------------------------------------------------------
# Node: generate product answer
# ---------------------------------------------------------------------------
def generate_product_answer_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""
    parsed = state.get("parsed_query") or {}
    products = state.get("products") or []

    # Clear clarification state — the customer gave us a real filter
    session.pending_clarification = None
    session.clarification_attempts = 0
    session.last_parsed = parsed
    session.state = ConversationState.IDLE
    session.last_products = products

    context = _build_product_context(products)

    llm = get_llm()
    chain = PRODUCT_ANSWER_PROMPT | llm

    try:
        resp = chain.invoke({
            "max_items": MAX_SUGGESTION_ITEMS,
            "question": question,
            "parsed": json.dumps(parsed, ensure_ascii=False),
            "context": context,
        })
        answer = safe_string(resp.content).strip()
    except Exception as error:
        logger.error("Product answer generation failed: %s", error)
        answer = _format_products_fallback(products)

    return {"answer": answer}


__all__ = [
    "understand_query_node",
    "clarify_node",
    "search_structured_node",
    "search_keyword_node",
    "search_semantic_node",
    "merge_results_node",
    "generate_product_answer_node",
]