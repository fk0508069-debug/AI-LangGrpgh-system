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


def _render_order(order: dict, tracking: str) -> str:
    """Render a professional order summary from the database document."""
    
    # 1. Extract embedded customer details directly from the order
    customer = order.get("customer", {})
    buyer_name = safe_string(customer.get("fullName"), "Customer")
    phone = safe_string(customer.get("phone"))
    
    address_parts = [
        safe_string(customer.get("address")),
        safe_string(customer.get("city")),
        safe_string(customer.get("state")),
        safe_string(customer.get("postalCode")),
    ]
    address = ", ".join(part for part in address_parts if part)

    # 2. Extract order meta
    status_val = safe_string(order.get("status"), "Not available").capitalize()
    order_date = get_order_date(order)
    expected = add_business_days(order_date, 5) if order_date else None

    items = order.get("items") or []
    if not isinstance(items, list):
        items = []

    # 3. Build the response
    lines = ["Here are the details of your order.", "", "**Order Summary**"]
    lines.append(f"• Tracking Number: {tracking}")
    lines.append(f"• Status: {status_val}")
    lines.append(f"• Customer Name: {buyer_name}")
    
    if phone:
        lines.append(f"• Contact Number: {phone}")
    if address:
        lines.append(f"• Shipping Address: {address}")
    if order_date:
        lines.append(f"• Placed on: {order_date.strftime('%B %d, %Y')}")
        
    if expected and status_val.lower() not in ["cancelled", "canceled"]:
        lines.append(f"• Expected Arrival: {expected.strftime('%B %d, %Y')}")
        lines.append("• Delivery Window: Up to 5 business days")

    # 4. Add Item Details
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
        
        # Financial Summary
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

    # 5. Closing Status Message
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
                f"**{tracking}**. Please double-check the number and try again."
            )
        }

    session.tracking_number = tracking
    session.order_data = order
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
        session.order_data = order

    question = (state.get("original_question") or "").lower().strip()

    from app.nodes.routing import (  # local import to avoid cycles
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