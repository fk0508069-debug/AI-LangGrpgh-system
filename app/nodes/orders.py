"""Order tracking, cancellation, follow-up, and confirmation nodes."""

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


def _buyer_info(order: dict) -> tuple[str, str]:
    user_id = order.get("userId")
    if not user_id:
        return "", ""
    try:
        user = mongodb.get_user_by_id(user_id)
        if not user:
            return "", ""
        return (
            safe_string(user.get("name") or user.get("fullName")),
            safe_string(
                user.get("phone") or user.get("phoneNumber") or user.get("contact")
            ),
        )
    except Exception:
        logger.exception("User lookup failed.")
        return "", ""


def _render_order(order: dict, tracking: str) -> str:
    buyer_name, phone = _buyer_info(order)
    order_id = safe_string(order.get("order_id") or order.get("orderId"))
    status_val = safe_string(order.get("status"), "Not available")
    order_date = get_order_date(order)
    expected = add_business_days(order_date, 5) if order_date else None

    items = order.get("items") or []
    if not isinstance(items, list):
        items = []

    lines = ["Here are the details of your order 📦", "", "**Order Summary**"]
    lines.append(f"• Tracking number: {tracking}")
    if order_id:
        lines.append(f"• Order ID: {order_id}")
    lines.append(f"• Status: {status_val}")
    if buyer_name:
        lines.append(f"• Buyer: {buyer_name}")
    if phone:
        lines.append(f"• Contact: {phone}")
    if order_date:
        lines.append(f"• Placed on: {order_date.strftime('%B %d, %Y')}")
    if expected:
        lines.append(f"• Expected arrival: {expected.strftime('%B %d, %Y')}")
        lines.append("• Delivery window: up to 5 business days")

    if items:
        lines += ["", "**Items**"]
        grand = 0.0
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            name = safe_string(item.get("name"), "Item")
            qty = safe_int(item.get("quantity") or item.get("qty") or 1, 1)
            unit = safe_float(item.get("price"))
            line = unit * qty
            grand += line
            lines.append(f"{idx}. **{name}**")
            lines.append(f"   • Quantity: {qty}")
            lines.append(f"   • Unit price: Rs. {format_money(unit)}")
            lines.append(f"   • Line total: Rs. {format_money(line)}")
        stored_total = order.get("total")
        total = safe_float(stored_total) if stored_total is not None else grand
        if total > 0:
            lines += ["", f"**Total: Rs. {format_money(total)}**"]
    else:
        lines += ["", "_No item details are available for this order._"]

    if expected:
        lines += [
            "",
            f"Your order is on track and should arrive around "
            f"**{expected.strftime('%B %d, %Y')}**. 💙",
        ]
    else:
        lines += ["", "Your order is being processed. 💙"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def order_tracking_node(state: GraphState) -> dict:
    session = state["session"]
    tracking = state.get("tracking_number") or ""
    order = mongodb.find_order_by_tracking_number(tracking)

    if not order:
        return {
            "answer": (
                f"I couldn't find an order with tracking number "
                f"{tracking}. Please double-check and try again. 😊"
            )
        }

    session.tracking_number = tracking
    session.order_data = order
    session.state = ConversationState.ORDER_FOLLOWUP
    return {"answer": _render_order(order, tracking)}


def order_tracking_prompt_node(state: GraphState) -> dict:
    return {
        "answer": "Sure! Please share your tracking number and I'll look it up for you. 😊"
    }


def order_followup_node(state: GraphState) -> dict:
    session = state["session"]
    tracking = session.tracking_number
    order = session.order_data

    if not tracking:
        return {
            "answer": (
                "I don't have an active order in this session. "
                "Could you share the tracking number? 📦"
            )
        }

    if not order:
        order = mongodb.find_order_by_tracking_number(tracking)
        if not order:
            return {
                "answer": f"I couldn't find order **{tracking}** anymore. Please try again."
            }
        session.order_data = order

    question = (state.get("original_question") or "").lower().strip()

    from app.nodes.routing import (  # local import to avoid cycles
        ORDER_DELIVERY_PHRASES,
        ORDER_STATUS_PHRASES,
        TRACKING_FOLLOWUP_PHRASES,
        _contains_phrase,
    )

    if _contains_phrase(question, TRACKING_FOLLOWUP_PHRASES):
        return {"answer": f"Your tracking number is **{tracking}**. 📦"}

    if _contains_phrase(question, ORDER_STATUS_PHRASES):
        status_val = safe_string(order.get("status"), "not available")
        return {"answer": f"Your order is currently marked as **{status_val}**. 😊"}

    if _contains_phrase(question, ORDER_DELIVERY_PHRASES):
        order_date = get_order_date(order)
        if order_date:
            expected = add_business_days(order_date, 5)
            return {
                "answer": (
                    f"Your expected delivery date is "
                    f"**{expected.strftime('%B %d, %Y')}**. 😊"
                )
            }
        return {
            "answer": (
                "I can see your order, but the placement date is "
                "missing so I can't estimate delivery."
            )
        }

    return {
        "answer": "I can help with that — just tell me what you'd like to know about the order. 📦"
    }


def cancel_request_node(state: GraphState) -> dict:
    session = state["session"]
    tracking = state.get("tracking_number") or session.tracking_number

    if not tracking:
        return {
            "answer": (
                "I can help with that. Please send your order's tracking number "
                "so I can check whether it is still eligible for cancellation. 📦"
            )
        }

    order = session.order_data if session.tracking_number == tracking else None
    if not order:
        order = mongodb.find_order_by_tracking_number(tracking)
    if not order:
        return {
            "answer": (
                f"I couldn't find an order with tracking number {tracking}. "
                "Please double-check it and try again."
            )
        }

    session.tracking_number = tracking
    session.order_data = order

    status_val = (
        safe_string(order.get("status"), "").lower().replace("-", "_").replace(" ", "_")
    )
    if status_val not in CANCELLABLE_STATUSES:
        return {
            "answer": (
                f"This order is currently **{safe_string(order.get('status'), 'not available')}** "
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
        return {"answer": "Understood. Your order has not been cancelled. 😊"}

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
    status_val = (
        safe_string(order.get("status"), "").lower().replace("-", "_").replace(" ", "_")
    )

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
                "You will receive an update if a refund is applicable."
            )
        }

    session.state = ConversationState.IDLE
    return {
        "answer": "I could not cancel the order right now. Please contact customer support."
    }


__all__ = [
    "order_tracking_node",
    "order_tracking_prompt_node",
    "order_followup_node",
    "cancel_request_node",
    "cancel_confirmation_node",
]