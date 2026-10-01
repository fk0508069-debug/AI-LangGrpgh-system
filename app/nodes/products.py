"""Product pipeline: understand → clarify → search → merge → generate."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple

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
VOCAB_CACHE_TTL = timedelta(minutes=30)
HISTORY_TAIL = 6

VOCAB_FIELDS = [
    "category", "subcategory", "subsubcategory", "brand",
    "color", "size", "material", "gender",
]

REQUIRED_SLOTS: Tuple[str, ...] = ("category", "price_range")

_SLOT_QUESTIONS: Dict[str, str] = {
    "category": (
        "Sure! What type of product are you looking for? "
        "For example: watch, shoes, shirt, mobile, bag."
    ),
    "price_range": (
        "Got it 👍 What's your price range? "
        "You can say something like \"under 1000\" or \"between 500 and 2000\"."
    ),
}

# Phrases that mean "skip the price question".
_ANY_PRICE_TOKENS = (
    "any price", "any budget",
    "doesn't matter", "doesnt matter", "dont matter", "don't matter",
    "dont care", "don't care",
    "no budget", "no price",
    "flexible", "anything", "whatever",
    "i dont know", "i don't know", "idk",
    "just show me", "show me one", "show me anything", "show me something",
    "surprise me", "your choice", "you pick", "up to you",
    "no preference",
)

_NUMBER_RE = re.compile(r"\d[\d,]*")

# Text fields we OR-search and rank against.
_TEXT_FIELDS = (
    "name", "description", "category", "subcategory",
    "subsubcategory", "brand", "color", "tags",
)

# Words that carry no product meaning.
_STOPWORDS = frozenset({
    "a", "an", "the", "me", "i", "want", "need", "show", "find", "get",
    "for", "of", "with", "and", "or", "to", "in", "on", "some", "any",
    "product", "products", "item", "items", "under", "below", "over",
    "between", "price", "range", "rs", "rupees", "pkr", "please",
})


# ---------------------------------------------------------------------------
# Singletons / caches
# ---------------------------------------------------------------------------
_llm_client = None


def _llm():
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm()
    return _llm_client


_vocab_cache: Optional[Dict[str, List[str]]] = None
_vocab_cache_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _history_as_dicts(session, tail: Optional[int] = None) -> List[Dict[str, str]]:
    from langchain_core.messages import AIMessage

    msgs = session.history
    if tail is not None:
        msgs = list(islice(reversed(msgs), tail))
        msgs.reverse()

    return [
        {
            "role": "assistant" if isinstance(m, AIMessage) else "user",
            "content": str(m.content),
        }
        for m in msgs
    ]


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


def _get_vocabulary() -> Dict[str, List[str]]:
    global _vocab_cache, _vocab_cache_time
    now = datetime.now()
    if (
        _vocab_cache is not None
        and _vocab_cache_time is not None
        and now - _vocab_cache_time < VOCAB_CACHE_TTL
    ):
        return _vocab_cache
    try:
        vocab = _fetch_vocabulary()
    except Exception:
        logger.exception("Vocab fetch failed.")
        vocab = {}
    _vocab_cache = vocab
    _vocab_cache_time = now
    return vocab


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
# Slot / price helpers
# ---------------------------------------------------------------------------
def _user_skipped_price(text: str) -> bool:
    lowered = (text or "").lower()
    return any(tok in lowered for tok in _ANY_PRICE_TOKENS)


def _missing_slot(parsed: Dict[str, Any], raw_text: str = "") -> Optional[str]:
    """Pure: which slot is still unfilled (or None)."""
    if not parsed.get("category") and not parsed.get("product_name"):
        return "category"

    has_price = (
        parsed.get("min_price") is not None
        or parsed.get("max_price") is not None
        or parsed.get("price_any") is True
        or _user_skipped_price(raw_text)
    )
    if not has_price:
        return "price_range"

    return None


def _resolve_missing_slot(
    parsed: Dict[str, Any],
    raw_text: str,
    session: Any,
) -> Optional[str]:
    """Session-aware variant: stop asking for price after one retry."""
    missing = _missing_slot(parsed, raw_text)
    if missing != "price_range":
        return missing

    if getattr(session, "pending_slot", None) == "price_range":
        parsed["price_any"] = True
        return None

    return "price_range"


def _extract_bare_price(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Bare number → (min, max). Single number is treated as an upper bound."""
    nums: List[float] = []
    for raw in _NUMBER_RE.findall(text or ""):
        val = safe_float(raw.replace(",", ""))
        if val is not None and val > 0:
            nums.append(val)
    if not nums:
        return None, None
    if len(nums) == 1:
        return None, nums[0]
    return min(nums), max(nums)


# ---------------------------------------------------------------------------
# Smart search — tokenize, OR-search Mongo, rank in Python
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [
        w for w in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(w) >= 2 and w not in _STOPWORDS
    ]


def _extract_search_tokens(parsed: Dict[str, Any]) -> List[str]:
    """Every meaningful word the user implied, deduped, order preserved."""
    raw: List[str] = []
    for key in ("product_name", "category", "subcategory", "subsubcategory",
                "brand", "color", "material"):
        v = parsed.get(key)
        if v:
            raw.append(str(v))
    kw = parsed.get("keywords") or []
    if isinstance(kw, list):
        raw.extend(str(k) for k in kw)

    seen: set = set()
    tokens: List[str] = []
    for chunk in raw:
        for w in _tokenize(chunk):
            if w not in seen:
                seen.add(w)
                tokens.append(w)
    return tokens


def _score_product(
    product: Dict[str, Any],
    tokens: List[str],
    parsed: Dict[str, Any],
) -> float:
    """How well does this product match what the user asked for?"""
    if not tokens:
        return 0.0

    name = str(product.get("name") or "").lower()
    desc = str(product.get("description") or "").lower()
    cat = str(product.get("category") or "").lower()
    subcat = str(product.get("subcategory") or "").lower()
    brand = str(product.get("brand") or "").lower()
    color = str(product.get("color") or "").lower()

    score = 0.0
    hits = 0

    # Whole-phrase bonus: "desk lamp" appears verbatim in the name.
    pname = str(
        parsed.get("product_name") or parsed.get("category") or ""
    ).lower().strip()
    if pname and pname in name:
        score += 8.0

    for tok in tokens:
        matched = False
        if tok in name:
            score += 3.0
            matched = True
        if tok in cat or tok in subcat:
            score += 2.0
            matched = True
        if tok in brand:
            score += 1.5
            matched = True
        if tok in color:
            score += 1.0
            matched = True
        if tok in desc:
            score += 0.5
            matched = True
        if matched:
            hits += 1

    # Coverage bonus: how many of the user's words landed somewhere.
    score += 4.0 * (hits / len(tokens))

    # Small in-stock nudge.
    try:
        if float(product.get("stock") or 0) > 0:
            score += 0.3
    except (TypeError, ValueError):
        pass

    return score


# ---------------------------------------------------------------------------
# Node: understand query
# ---------------------------------------------------------------------------
def understand_query_node(state: GraphState) -> dict:
    session = state["session"]
    question = (state.get("original_question") or "").strip()

    pending = session.pending_clarification
    pending_slot = getattr(session, "pending_slot", None)

    if pending:
        merged = f"Previous request: {pending}\nUser's clarification: {question}"
    else:
        merged = question

    vocab = _get_vocabulary()

    system_msg = QUERY_UNDERSTANDING_PROMPT.format_messages(
        vocabulary=json.dumps(vocab, ensure_ascii=False),
        question=merged,
    )[0]

    messages = [system_msg]

    if pending:
        for turn in _history_as_dicts(session, tail=HISTORY_TAIL):
            messages.append(turn)

    messages.append({"role": "user", "content": merged})

    try:
        content = safe_string(_llm().invoke(messages).content).strip()
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

    # Merge with the previous parse when mid-clarification.
    if pending and session.last_parsed:
        for key in (*VOCAB_FIELDS, "min_price", "max_price", "sort",
                    "in_stock", "product_name", "price_any"):
            if parsed.get(key) in (None, [], ""):
                parsed[key] = session.last_parsed.get(key)
        prior_kw = session.last_parsed.get("keywords") or []
        new_kw = parsed.get("keywords") or []
        if isinstance(prior_kw, list) and isinstance(new_kw, list):
            parsed["keywords"] = list(dict.fromkeys([*prior_kw, *new_kw]))

    # Inherit context when refining a previous product request even if the
    # clarification thread was cleared (e.g. after a "no results" answer).
    if not pending and session.last_parsed:
        last = session.last_parsed
        looks_like_refinement = (
            (parsed.get("min_price") is not None
             or parsed.get("max_price") is not None
             or parsed.get("price_any") is True
             or _user_skipped_price(question))
            and not parsed.get("category")
            and not parsed.get("product_name")
        )
        if looks_like_refinement:
            inherited = False
            for key in ("category", "subcategory", "subsubcategory",
                        "product_name", "brand", "color"):
                if not parsed.get(key) and last.get(key):
                    parsed[key] = last[key]
                    inherited = True
            if isinstance(last.get("keywords"), list):
                prior_kw = last.get("keywords")
                new_kw = parsed.get("keywords") or []
                parsed["keywords"] = list(dict.fromkeys([*prior_kw, *new_kw]))
            if inherited:
                logger.info(
                    "understand: inherited context | category=%s product=%s",
                    parsed.get("category"), parsed.get("product_name"),
                )

    # Post-process: bare-number reply to our price question.
    if pending_slot == "price_range":
        lo, hi = parsed.get("min_price"), parsed.get("max_price")
        if lo is None and hi is None:
            lo, hi = _extract_bare_price(question)
            if lo is not None:
                parsed["min_price"] = lo
            if hi is not None:
                parsed["max_price"] = hi
        elif lo is not None and hi is not None and lo == hi:
            parsed["min_price"] = None  # "1000" → "under 1000"
        if _user_skipped_price(question):
            parsed["price_any"] = True

    # Post-process: short word reply to our category question.
    if pending_slot == "category" and not parsed.get("category"):
        clean = question.strip()
        if 0 < len(clean) <= 40 and re.fullmatch(r"[A-Za-z][A-Za-z\s\-']*", clean):
            parsed["category"] = clean.title()
            kws = parsed.setdefault("keywords", [])
            if isinstance(kws, list):
                kws.append(clean.lower())

    if _resolve_missing_slot(parsed, merged, session):
        parsed["needs_clarification"] = True

    return {"parsed_query": parsed}


# ---------------------------------------------------------------------------
# Node: clarify
# ---------------------------------------------------------------------------
def clarify_node(state: GraphState) -> dict:
    session = state["session"]
    question = (state.get("original_question") or "").strip()
    parsed = state.get("parsed_query") or {}

    from app.nodes.routing import is_garbage_input, is_irrelevant_reply

    if session.pending_clarification and (
        is_garbage_input(question) or is_irrelevant_reply(question)
    ):
        session.clarification_attempts += 1
        if session.clarification_attempts >= 2:
            session.clarification_attempts = 0
            return {
                "answer": (
                    "I'm not sure I understood that. 😊 Could you describe "
                    "what you're looking for — for example, a category, a "
                    "product type, or your budget?"
                )
            }
        return {
            "answer": (
                "Sorry, I didn't understand that. Could you tell me a bit more "
                "about what you'd like to shop for?"
            )
        }

    missing = _missing_slot(parsed, question)

    if missing:
        clarification = _SLOT_QUESTIONS[missing]
        session.pending_slot = missing
    else:
        clarification = parsed.get("clarification_question")
        if not clarification:
            vocab = _get_vocabulary()
            history_text = "\n".join(
                f"{t['role']}: {t['content']}"
                for t in _history_as_dicts(session, tail=8)
            )
            try:
                chain = CLARIFICATION_PROMPT | _llm()
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
        session.pending_slot = None

    if session.pending_clarification:
        session.pending_clarification = f"{session.pending_clarification} | {question}"
    else:
        session.pending_clarification = question

    session.last_parsed = parsed
    session.state = ConversationState.AWAITING_CLARIFICATION
    return {"answer": clarification, "needs_clarification": True}


# ---------------------------------------------------------------------------
# Nodes: search
# ---------------------------------------------------------------------------
def search_structured_node(state: GraphState) -> dict:
    parsed = state.get("parsed_query") or {}
    tokens = _extract_search_tokens(parsed)

    # --- hard filters: price + stock ---------------------------------------
    base: Dict[str, Any] = {}
    price_q: Dict[str, Any] = {}
    if parsed.get("min_price") is not None:
        price_q["$gte"] = safe_float(parsed["min_price"])
    if parsed.get("max_price") is not None:
        price_q["$lte"] = safe_float(parsed["max_price"])
    if price_q:
        base["price"] = price_q
    if parsed.get("in_stock") is True:
        base["stock"] = {"$gt": 0}

    # --- candidate pool ----------------------------------------------------
    candidates: List[Dict[str, Any]] = []
    try:
        if tokens:
            ors: List[Dict[str, Any]] = []
            for tok in tokens:
                pat = {"$regex": re.escape(tok), "$options": "i"}
                for f in _TEXT_FIELDS:
                    ors.append({f: pat})
            query: Dict[str, Any] = {**base, "$or": ors}
            fallback_query: Dict[str, Any] = {"$or": ors}
        else:
            query = dict(base)
            fallback_query = {}

        candidates = list(mongodb.products.find(query).limit(80))

        # If price narrowed us to nothing, drop the price clause and retry.
        if not candidates and price_q:
            candidates = list(mongodb.products.find(fallback_query).limit(80))
            if candidates:
                logger.info(
                    "search: relaxed price filter (0 strict matches for %s)",
                    price_q,
                )
    except Exception:
        logger.exception("Structured search failed.")
        candidates = []

    if not candidates:
        return {"structured_products": []}

    # --- rank in Python (single pass) --------------------------------------
    scored = [(_score_product(p, tokens, parsed), p) for p in candidates]

    if tokens:
        scored = [(s, p) for s, p in scored if s > 0]

    if not scored:
        return {"structured_products": []}

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [p for _, p in scored[:MAX_SUGGESTION_ITEMS]]

    logger.info(
        "search: %d candidates → top score %.2f → %d results",
        len(candidates), scored[0][0], len(results),
    )
    return {"structured_products": results}


def search_keyword_node(state: GraphState) -> dict:
    parsed = state.get("parsed_query") or {}
    keywords = parsed.get("keywords") or []
    product_name = parsed.get("product_name")

    terms: List[str] = []
    if product_name:
        terms.append(str(product_name))
    if isinstance(keywords, list):
        terms.extend(keywords)
    terms = list(dict.fromkeys(t.strip() for t in terms if str(t).strip()))
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
    parsed = state.get("parsed_query") or {}

    has_category = bool(parsed.get("category") or parsed.get("product_name"))
    has_price = (
        parsed.get("min_price") is not None
        or parsed.get("max_price") is not None
        or parsed.get("price_any") is True
    )
    if has_category and has_price:
        return {"documents": []}

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
def _bulk_get_products(ids: List[str]) -> List[Dict[str, Any]]:
    from bson import ObjectId

    oids: List[Any] = []
    for i in ids:
        try:
            oids.append(ObjectId(i))
        except Exception:
            oids.append(i)
    try:
        return list(mongodb.products.find({"_id": {"$in": oids}}))
    except Exception:
        logger.exception("Bulk product fetch failed.")
        return []


def merge_results_node(state: GraphState) -> dict:
    structured = state.get("structured_products") or []
    keyword = state.get("keyword_products") or []
    documents = state.get("documents") or []

    merged: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(product: Dict[str, Any]) -> bool:
        pid = str(product.get("_id"))
        if pid in seen:
            return False
        seen.add(pid)
        merged.append(product)
        return len(merged) >= MAX_SUGGESTION_ITEMS

    for p in structured:
        if _add(p):
            return {"products": merged}

    if len(merged) < MAX_SUGGESTION_ITEMS:
        for p in keyword:
            if _add(p):
                return {"products": merged}

    if len(merged) < MAX_SUGGESTION_ITEMS:
        sem_ids = list(dict.fromkeys(
            m for m in (
                (d.metadata or {}).get("mongo_id") for d in documents
            )
            if m and m not in seen
        ))
        if sem_ids:
            for p in _bulk_get_products(sem_ids[:MAX_SUGGESTION_ITEMS]):
                if _add(p):
                    break

    return {"products": merged}


# ---------------------------------------------------------------------------
# Node: generate product answer
# ---------------------------------------------------------------------------
def generate_product_answer_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""
    parsed = state.get("parsed_query") or {}
    products = state.get("products") or []

    session.last_parsed = parsed

    if products:
        session.pending_clarification = None
        session.pending_slot = None
        session.clarification_attempts = 0
        session.state = ConversationState.IDLE
        session.last_products = products
    else:
        # Empty result: keep the thread alive and ANCHOR it to the category
        # so a follow-up like "under 2000" doesn't lose "laptop".
        session.pending_slot = None
        session.state = ConversationState.AWAITING_CLARIFICATION

        prior = session.last_parsed or {}
        anchor = (
            parsed.get("category")
            or prior.get("category")
            or parsed.get("product_name")
            or prior.get("product_name")
            or ""
        )
        if anchor:
            session.pending_clarification = f"{anchor} — {question}"
        elif not session.pending_clarification:
            session.pending_clarification = question

    context = _build_product_context(products)

    # Optional: inject cheapest-in-category hint when nothing matched.
    if not products:
        category = (
            parsed.get("category")
            or (session.last_parsed or {}).get("category")
        )
        if category:
            try:
                cheapest = mongodb.cheapest_in_category(category)  # type: ignore[attr-defined]
            except AttributeError:
                cheapest = None
            except Exception:
                logger.exception("cheapest_in_category failed")
                cheapest = None
            if cheapest:
                context += (
                    f"\n\nCHEAPEST IN CATEGORY ({category}): "
                    f"Rs. {format_money(cheapest.get('price'))}"
                )

    try:
        chain = PRODUCT_ANSWER_PROMPT | _llm()
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