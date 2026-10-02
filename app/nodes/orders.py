"""Order tracking, cancellation, reorder, and confirmation nodes."""

from __future__ import annotations

import logging
import re

from app.services.helpers import (
    add_business_days,
    format_money,
    get_order_date,
    hours_since_order,
    safe_float,
    safe_int,
    safe_string,
)
from app.services.mongodb import mongodb
from app.state import ConversationState, GraphState

logger = logging.getLogger(__name__)

CONFIRMATION_YES = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|okay|ok|confirm|confirmed|please do|do it|go ahead)$",
    re.IGNORECASE,
)
CONFIRMATION_NO = re.compile(
    r"^(?:no|nope|nah|cancel that|don't|do not|never mind|nevermind)$",
    re.IGNORECASE,
)

CANCELLABLE_STATUSES = {"pending", "processing", "in_process", "placed", "confirmed"}


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _order_owner_id(order: dict) -> str | None:
    """Best-effort extraction of the order owner's customer id."""
    if not order:
        return None
    for key in ("customerId", "customer_id", "userId", "user_id"):
        v = order.get(key)
        if v:
            return str(v)
    customer = order.get("customer") or {}
    for key in ("id", "_id", "customerId", "customer_id"):
        v = customer.get(key)
        if v:
            return str(v)
    return None


def _authorized_for_order(state: GraphState, order: dict) -> bool:
    """True if caller owns the order OR no identity is set (dev / not enforced)."""
    session = state["session"]
    caller = session.customer_id
    owner = _order_owner_id(order)

    # No auth configured → do not block (bootstrap mode).
    if not caller or not owner:
        return True
    return str(caller) == str(owner)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_order(order: dict, tracking: str) -> str:
    """Render a professional, PII-masked order summary."""
    customer = order.get("customer") or {}
    raw_name = safe_string(customer.get("fullName"), "Customer")
    parts = raw_name.split()
    if len(parts) >= 2:
        masked_name = f"{parts[0]} {parts[-1][0]}."
    else:
        masked_name = raw_name

    phone = safe_string(customer.get("phone"))
    masked_phone = f"***{phone[-4:]}" if len(phone) >= 4 else ""

    city = safe_string(customer.get("city"))
    state_ = safe_string(customer.get("state"))
    location = ", ".join(p for p in [city, state_] if p)

    status_val = safe_string(order.get("status"), "Not available").capitalize()
    order_date = get_order_date(order)
    expected = add_business_days(order_date, 5) if order_date else None

    items = order.get("items") or []
    if not isinstance(items, list):
        items = []

    lines = ["Here are the details of your order.", "", "**Order Summary**"]
    lines.append(f"• Tracking Number: {tracking}")
    lines.append(f"• Status: {status_val}")
    lines.append(f"• Customer: {masked_name}")
    if masked_phone:
        lines.append(f"• Contact: {masked_phone}")
    if location:
        lines.append(f"• Ship to: {location}")
    if order_date:
        lines.append(f"• Placed on: {order_date.strftime('%B %d, %Y')}")

    if expected and status_val.lower() not in ["cancelled", "canceled"]:
        lines.append(f"• Expected Arrival: {expected.strftime('%B %d, %Y')}")
        lines.append("• Delivery Window: Up to 5 business days")

    if items:
        lines += ["", "**Items Ordered**"]
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            name = safe_string(item.get("name"), "Item")
            sku = safe_string(item.get("sku"), "N/A")
            qty = safe_int(item.get("quantity"), 1)
            unit = safe_float(item.get("unitPrice"))
            line_total = safe_float(item.get("lineTotal"), unit * qty)

            lines.append(f"{idx}. **{name}**")
            lines.append(f"   • SKU: {sku}")
            lines.append(f"   • Quantity: {qty}")
            lines.append(f"   • Unit Price: Rs. {format_money(unit)}")
            lines.append(f"   • Line Total: Rs. {format_money(line_total)}")

        subtotal = safe_float(order.get("subtotal", 0))
        shipping = safe_float(order.get("shipping", 0))
        total = safe_float(order.get("total", 0))

        lines += ["", "**Order Total**"]
        if subtotal > 0:
            lines.append(f"• Subtotal: Rs. {format_money(subtotal)}")
        if shipping >= 0:
            lines.append(f"• Shipping: Rs. {format_money(shipping)}")
        lines.append(f"**• Total: Rs. {format_money(total)}**")
    else:
        lines += ["", "_No item details are available for this order._"]

    if status_val.lower() in ["cancelled", "canceled"]:
        lines += ["", "This order has been cancelled."]
    elif expected:
        lines += [
            "",
            f"Your order is on track and should arrive around "
            f"**{expected.strftime('%B %d, %Y')}**.",
        ]
    else:
        lines += ["", "Your order is currently being processed."]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
def order_tracking_node(state: GraphState) -> dict:
    session = state["session"]
    tracking = state.get("tracking_number") or ""

    order = mongodb.find_order_by_tracking_number(tracking)
    if not order:
        return {
            "answer": (
                f"I couldn't find an order with tracking number "
                f"**{tracking}**. Please double-check the number and try again."
            )
        }

    if not _authorized_for_order(state, order):
        # Do NOT reveal existence of another user's order.
        return {
            "answer": (
                f"I couldn't find an order with tracking number "
                f"**{tracking}**. Please double-check the number and try again."
            )
        }

    session.tracking_number = tracking
    session.order_data = order
    session.last_order_id = str(order.get("_id") or order.get("id") or "")
    session.last_order_items = order.get("items") or []
    session.last_order_total = safe_float(order.get("total"))
    session.state = ConversationState.ORDER_FOLLOWUP

    return {"answer": _render_order(order, tracking)}


def order_tracking_prompt_node(state: GraphState) -> dict:
    return {
        "answer": "Please provide your tracking number so I can look up your order."
    }


def order_followup_node(state: GraphState) -> dict:
    session = state["session"]
    tracking = session.tracking_number
    order = session.order_data

    if not tracking:
        return {
            "answer": (
                "I don't have an active order in this session. "
                "Could you please provide the tracking number?"
            )
        }

    if not order:
        order = mongodb.find_order_by_tracking_number(tracking)
        if not order:
            return {
                "answer": f"I couldn't find order **{tracking}** anymore. Please try again."
            }
        if not _authorized_for_order(state, order):
            return {
                "answer": f"I couldn't find order **{tracking}** anymore. Please try again."
            }
        session.order_data = order

    question = (state.get("original_question") or "").lower().strip()

    from app.nodes.routing import (
        ORDER_DELIVERY_PHRASES,
        ORDER_STATUS_PHRASES,
        TRACKING_FOLLOWUP_PHRASES,
        _contains_phrase,
    )

    if _contains_phrase(question, TRACKING_FOLLOWUP_PHRASES):
        return {"answer": f"Your tracking number is **{tracking}**."}

    if _contains_phrase(question, ORDER_STATUS_PHRASES):
        status_val = safe_string(order.get("status"), "not available").capitalize()
        return {"answer": f"Your order is currently marked as **{status_val}**."}

    if _contains_phrase(question, ORDER_DELIVERY_PHRASES):
        status_val = safe_string(order.get("status"), "").lower()
        if status_val in ["cancelled", "canceled"]:
            return {"answer": "This order has been cancelled and will not be delivered."}

        order_date = get_order_date(order)
        if order_date:
            expected = add_business_days(order_date, 5)
            return {
                "answer": (
                    f"Your expected delivery date is "
                    f"**{expected.strftime('%B %d, %Y')}**."
                )
            }
        return {
            "answer": (
                "I can see your order, but the placement date is "
                "missing so I can't estimate the delivery date."
            )
        }

    return {
        "answer": "I can help with that. Please tell me what you'd like to know about the order."
    }


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------
def cancel_request_node(state: GraphState) -> dict:
    session = state["session"]
    tracking = state.get("tracking_number") or session.tracking_number

    if not tracking:
        return {
            "answer": (
                "I can help with that. Please send your order's tracking number "
                "so I can check whether it is still eligible for cancellation."
            )
        }

    order = session.order_data if session.tracking_number == tracking else None
    if not order:
        order = mongodb.find_order_by_tracking_number(tracking)
    if not order:
        return {
            "answer": (
                f"I couldn't find an order with tracking number **{tracking}**. "
                "Please double-check it and try again."
            )
        }

    if not _authorized_for_order(state, order):
        return {
            "answer": (
                f"I couldn't find an order with tracking number **{tracking}**. "
                "Please double-check it and try again."
            )
        }

    session.tracking_number = tracking
    session.order_data = order

    status_val = safe_string(order.get("status"), "").lower().replace("-", "_").replace(" ", "_")

    if status_val in ["cancelled", "canceled"]:
        return {"answer": "This order has already been cancelled."}

    if status_val not in CANCELLABLE_STATUSES:
        return {
            "answer": (
                f"This order is currently **{safe_string(order.get('status'), 'not available').capitalize()}** "
                "and cannot be cancelled here. Please contact customer support for help."
            )
        }

    age_hours = hours_since_order(order)
    if age_hours is None:
        return {
            "answer": (
                "I found the order, but its placement time is missing. "
                "Please contact customer support to request cancellation."
            )
        }
    if age_hours >= 24:
        return {
            "answer": (
                f"This order was placed {age_hours / 24:.1f} days ago, so it is past "
                "the 24-hour cancellation window. Please contact customer support."
            )
        }

    session.state = ConversationState.AWAITING_CANCEL_CONFIRMATION
    return {
        "answer": (
            f"Your order **{tracking}** is still within the 24-hour cancellation window.\n\n"
            "⚠️ **Do you really want to cancel this order?**\n"
            "Reply **yes** to confirm or **no** to keep it."
        )
    }


def cancel_confirmation_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""
    normalized = re.sub(r"[.!?]+$", "", question.strip()).lower()

    if CONFIRMATION_NO.fullmatch(normalized):
        session.state = ConversationState.ORDER_FOLLOWUP
        return {"answer": "Understood. Your order has not been cancelled."}

    if not CONFIRMATION_YES.fullmatch(normalized):
        return {"answer": "Please reply **yes** to cancel the order or **no** to keep it."}

    tracking = session.tracking_number
    order = session.order_data

    if not tracking or not order:
        session.state = ConversationState.IDLE
        return {
            "answer": "I could not verify the order. Please contact customer support."
        }

    age_hours = hours_since_order(order)
    status_val = safe_string(order.get("status"), "").lower().replace("-", "_").replace(" ", "_")

    if age_hours is None or age_hours >= 24 or status_val not in CANCELLABLE_STATUSES:
        session.state = ConversationState.IDLE
        return {
            "answer": (
                "The order is no longer eligible for cancellation. "
                "Please contact customer support."
            )
        }

    if mongodb.cancel_order(tracking):
        order["status"] = "cancelled"
        session.state = ConversationState.ORDER_FOLLOWUP
        return {
            "answer": (
                "✅ **Order cancelled successfully**\n\n"
                f"Tracking number: **{tracking}**\n"
                "Thanks for considering us please remmember us again  "
            )
        }

    session.state = ConversationState.IDLE
    return {
        "answer": "I could not cancel the order right now. Please contact customer support."
    }


# ---------------------------------------------------------------------------
# Reorder
# ---------------------------------------------------------------------------
def reorder_request_node(state: GraphState) -> dict:
    session = state["session"]

    order = session.order_data
    if not order and session.tracking_number:
        order = mongodb.find_order_by_tracking_number(session.tracking_number)

    if not order:
        return {
            "answer": (
                "I don't have a previous order in this session. "
                "Please share your tracking number and I'll look it up."
            )
        }

    if not _authorized_for_order(state, order):
        return {
            "answer": (
                "I don't have a previous order I can reorder for you. "
                "Please share your tracking number."
            )
        }

    items = order.get("items") or []
    if not items:
        return {"answer": "That order has no items I can reorder."}

    # Stock check — TODO: MONGODB — implement `sku_in_stock(sku, qty)`
    unavailable: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sku = safe_string(item.get("sku"))
        qty = safe_int(item.get("quantity"), 1)
        try:
            in_stock = mongodb.sku_in_stock(sku, qty)  # type: ignore[attr-defined]
        except AttributeError:
            # method not yet implemented — skip the check
            in_stock = True
        if not in_stock:
            unavailable.append(safe_string(item.get("name"), sku or "item"))

    if unavailable:
        return {
            "answer": (
                "I can't reorder right now — these items are out of stock:\n"
                + "\n".join(f"• {n}" for n in unavailable)
            )
        }

    total = safe_float(order.get("total"), 0)
    lines = ["I can reorder the items from your previous order:", ""]
    for item in items:
        if not isinstance(item, dict):
            continue
        name = safe_string(item.get("name"), "Item")
        qty = safe_int(item.get("quantity"), 1)
        unit = safe_float(item.get("unitPrice"))
        lines.append(f"• {name} × {qty} — Rs. {format_money(unit * qty)}")

    lines += [
        "",
        f"Total: Rs. {format_money(total)}",
        "",
        "Would you like me to place this order again?",
    ]

    session.pending_reorder = {
        "items": items,
        "total": total,
        "shipping": safe_float(order.get("shipping"), 0),
        "source_order_id": str(order.get("_id") or order.get("id") or ""),
    }
    session.state = ConversationState.AWAITING_REORDER_CONFIRMATION

    return {"answer": "\n".join(lines)}


def reorder_confirmation_node(state: GraphState) -> dict:
    session = state["session"]
    question = state.get("original_question") or ""
    normalized = re.sub(r"[.!?]+$", "", question.strip()).lower()

    from app.nodes.routing import REORDER_CONFIRM_NO, REORDER_CONFIRM_YES

    if REORDER_CONFIRM_NO.fullmatch(normalized):
        session.pending_reorder = None
        session.state = ConversationState.ORDER_FOLLOWUP
        return {"answer": "Understood — I won't reorder it."}

    if not REORDER_CONFIRM_YES.fullmatch(normalized):
        return {"answer": "Please reply **yes** to place the reorder or **no** to cancel."}

    pending = session.pending_reorder
    if not pending:
        session.state = ConversationState.IDLE
        return {"answer": "I lost the reorder details. Please ask again."}

    # TODO: MONGODB — implement `create_order_from_items(items, shipping, source_order_id)`
    try:
        new_order = mongodb.create_order_from_items(  # type: ignore[attr-defined]
            items=pending["items"],
            shipping=pending.get("shipping", 0),
            source_order_id=pending.get("source_order_id"),
        )
    except AttributeError:
        logger.exception("mongodb.create_order_from_items is not implemented")
        session.state = ConversationState.IDLE
        return {
            "answer": (
                "Reorder is not available right now — the backend method "
                "`create_order_from_items` is not implemented. Please contact support."
            )
        }
    except Exception:
        logger.exception("Reorder failed")
        session.state = ConversationState.IDLE
        return {"answer": "I couldn't place the reorder. Please contact support."}

    if not new_order:
        session.state = ConversationState.IDLE
        return {"answer": "I couldn't place the reorder. Please contact support."}

    new_tracking = (
        new_order.get("trackingNumber")
        or new_order.get("tracking_number")
        or new_order.get("orderNumber")
        or "N/A"
    )

    session.pending_reorder = None
    session.state = ConversationState.ORDER_FOLLOWUP
    session.tracking_number = str(new_tracking)
    session.order_data = new_order
    session.last_order_id = str(new_order.get("_id") or new_order.get("id") or "")
    session.last_order_items = new_order.get("items") or []
    session.last_order_total = safe_float(new_order.get("total"))

    total = safe_float(new_order.get("total"), pending.get("total", 0))
    return {
        "answer": (
            "✅ **Order placed successfully**\n\n"
            f"New tracking number: **{new_tracking}**\n"
            f"Total: Rs. {format_money(total)}"
        )
    }


__all__ = [
    "order_tracking_node",
    "order_tracking_prompt_node",
    "order_followup_node",
    "cancel_request_node",
    "cancel_confirmation_node",
    "reorder_request_node",
    "reorder_confirmation_node",
]