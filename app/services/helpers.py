"""Pure helpers — no side effects, easy to test."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return safe_string(value, "N/A")


def add_business_days(start: datetime, days: int) -> datetime:
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def get_order_date(order: dict) -> Optional[datetime]:
    raw = (
        order.get("createdAt")
        or order.get("created_at")
        or order.get("orderDate")
        or order.get("order_date")
    )
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    logger.warning("Unable to parse order date: %s", raw)
    return None


def hours_since_order(order: dict) -> Optional[float]:
    order_date = get_order_date(order)
    if not order_date:
        return None
    now = datetime.now(order_date.tzinfo) if order_date.tzinfo else datetime.now()
    return max(0.0, (now - order_date).total_seconds() / 3600)