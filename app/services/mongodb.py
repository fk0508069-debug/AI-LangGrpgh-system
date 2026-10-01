"""MongoDB client — products, orders, users."""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import get_settings

logger = logging.getLogger(__name__)


# Both snake_case and camelCase keys we accept for tracking numbers.
_TRACKING_KEYS = ("tracking_number", "trackingNumber", "orderNumber", "order_number")


class _MongoDB:
    def __init__(self) -> None:
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._logged_init = False

    # ------------------------------------------------------------------ setup
    def _ensure(self) -> Database:
        if self._db is None:
            s = get_settings()
            logger.info(
                "Connecting to MongoDB | uri=%s | db=%s | collection=%s",
                s.mongodb_uri,
                s.mongodb_db,
                s.mongo_collection,
            )
            self._client = MongoClient(s.mongodb_uri, serverSelectionTimeoutMS=5000)
            self._db = self._client[s.mongodb_db]
            if not self._logged_init:
                try:
                    names = self._client.list_database_names()
                    logger.info("MongoDB reachable. Databases: %s", names)
                    logger.info(
                        "Using DB '%s'. Collections: %s",
                        s.mongodb_db,
                        self._db.list_collection_names(),
                    )
                    count = self.products.estimated_document_count()
                    logger.info(
                        "Collection '%s' contains ~%d documents",
                        s.mongo_collection,
                        count,
                    )
                except Exception as e:
                    logger.error("MongoDB connectivity check failed: %s", e)
                self._logged_init = True
        return self._db

    @property
    def products(self) -> Collection:
        s = get_settings()
        return self._ensure()[s.mongo_collection]

    @property
    def orders(self) -> Collection:
        return self._ensure()["orders"]

    @property
    def users(self) -> Collection:
        return self._ensure()["users"]

    # ---------------------------------------------------------------- products
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.products.find_one({"_id": ObjectId(product_id)})
        except (InvalidId, Exception):
            return None

    def find_products(
        self,
        query: Dict[str, Any],
        limit: int = 10,
        sort: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        logger.info("Mongo find | query=%s | limit=%d", query, limit)
        cursor = self.products.find(query)
        if sort:
            for field, direction in sort:
                cursor = cursor.sort(field, direction)
        results = list(cursor.limit(limit))
        logger.info("Mongo find returned %d docs", len(results))
        return results

    def distinct(self, collection: Collection, field: str) -> List[Any]:
        try:
            return collection.distinct(field)
        except Exception:
            logger.exception("distinct(%s) failed", field)
            return []

    def sku_in_stock(self, sku: str, qty: int = 1) -> bool:
        """Return True if the product with `sku` has at least `qty` units."""
        if not sku:
            return False
        try:
            doc = self.products.find_one({"sku": sku})
        except Exception:
            logger.exception("sku_in_stock: product lookup failed for sku=%s", sku)
            return False
        if not doc:
            return False
        try:
            return int(doc.get("stock") or 0) >= int(qty or 1)
        except (TypeError, ValueError):
            return False

    def cheapest_in_category(self, category: str) -> Optional[Dict[str, Any]]:
        """Return the cheapest in-stock product in the given category (case-insensitive)."""
        if not category:
            return None
        try:
            return self.products.find_one(
                {
                    "category": {"$regex": f"^{category.strip()}$", "$options": "i"},
                    "stock": {"$gt": 0},
                },
                sort=[("price", 1)],
            )
        except Exception:
            logger.exception("cheapest_in_category failed for %s", category)
            return None

    # ------------------------------------------------------------------ orders
    def find_order_by_tracking_number(self, tracking: str) -> Optional[Dict[str, Any]]:
        tracking = str(tracking).strip()
        if not tracking:
            return None

        # Try string match across all known keys.
        or_clauses: List[Dict[str, Any]] = [{k: tracking} for k in _TRACKING_KEYS]
        order = self.orders.find_one({"$or": or_clauses})
        if order:
            return order

        # Try integer form (legacy orders stored numeric tracking).
        try:
            numeric_tracking = int(tracking)
        except (ValueError, TypeError):
            return None
        or_clauses = [{k: numeric_tracking} for k in _TRACKING_KEYS]
        return self.orders.find_one({"$or": or_clauses})

    def find_order_by_id(self, order_id: Any) -> Optional[Dict[str, Any]]:
        if not order_id:
            return None
        try:
            return self.orders.find_one({"_id": ObjectId(str(order_id))})
        except (InvalidId, Exception):
            try:
                return self.orders.find_one({"_id": order_id})
            except Exception:
                return None

    def cancel_order(self, tracking: str) -> bool:
        """Mark an order as cancelled. Matches both snake_case and camelCase keys."""
        tracking = str(tracking).strip()
        if not tracking:
            return False

        update = {"$set": {"status": "cancelled"}}
        candidates: List[Any] = [tracking]
        try:
            candidates.append(int(tracking))
        except (ValueError, TypeError):
            pass

        for value in candidates:
            for key in _TRACKING_KEYS:
                result = self.orders.update_one({key: value}, update)
                if result.modified_count > 0 or result.matched_count > 0:
                    return True
        return False

    def create_order_from_items(
        self,
        items: List[Dict[str, Any]],
        shipping: float = 0.0,
        source_order_id: Optional[str] = None,
        customer: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new order document from a list of line items.

        `items` entries are expected to have at least: name, sku, quantity,
        unitPrice. `lineTotal` is recomputed. Returns the inserted document
        (with `_id` and a freshly generated tracking number).
        """
        if not items:
            return None

        normalized: List[Dict[str, Any]] = []
        subtotal = 0.0

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                qty = int(item.get("quantity") or 1)
            except (TypeError, ValueError):
                qty = 1
            try:
                unit = float(item.get("unitPrice") or 0)
            except (TypeError, ValueError):
                unit = 0.0
            line_total = round(unit * qty, 2)
            subtotal += line_total
            normalized.append(
                {
                    "name": item.get("name"),
                    "sku": item.get("sku"),
                    "quantity": qty,
                    "unitPrice": unit,
                    "lineTotal": line_total,
                }
            )

        if not normalized:
            return None

        try:
            shipping_val = float(shipping or 0)
        except (TypeError, ValueError):
            shipping_val = 0.0

        now = _dt.datetime.utcnow()
        tracking = f"ORD-{now.year}-{uuid.uuid4().hex[:8].upper()}"

        doc: Dict[str, Any] = {
            "trackingNumber": tracking,
            "tracking_number": tracking,  # write both so legacy readers work
            "status": "placed",
            "customer": customer or {},
            "items": normalized,
            "subtotal": round(subtotal, 2),
            "shipping": round(shipping_val, 2),
            "total": round(subtotal + shipping_val, 2),
            "placedAt": now,
            "createdAt": now,
        }
        if source_order_id:
            doc["sourceOrderId"] = source_order_id

        try:
            result = self.orders.insert_one(doc)
        except Exception:
            logger.exception("create_order_from_items: insert failed")
            return None

        doc["_id"] = result.inserted_id
        return doc

    # ------------------------------------------------------------------- users
    def get_user_by_id(self, user_id: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(user_id, str):
                return self.users.find_one({"_id": ObjectId(user_id)})
            return self.users.find_one({"_id": user_id})
        except (InvalidId, Exception):
            return None


mongodb = _MongoDB()