"""MongoDB client — products, orders, users."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import get_settings

logger = logging.getLogger(__name__)


class _MongoDB:
    def __init__(self) -> None:
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._logged_init = False

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

    # ---- Products ----
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

    # ---- Orders ----
    def find_order_by_tracking_number(self, tracking: str) -> Optional[Dict[str, Any]]:
        tracking = str(tracking).strip()
        order = self.orders.find_one(
            {"$or": [{"tracking_number": tracking}, {"trackingNumber": tracking}]}
        )
        if order:
            return order
        try:
            numeric_tracking = int(tracking)
            return self.orders.find_one(
                {
                    "$or": [
                        {"tracking_number": numeric_tracking},
                        {"trackingNumber": numeric_tracking},
                    ]
                }
            )
        except (ValueError, TypeError):
            return None

    def cancel_order(self, tracking: str) -> bool:
        tracking = str(tracking).strip()
        result = self.orders.update_one(
            {"tracking_number": tracking},
            {"$set": {"status": "cancelled"}},
        )
        if result.matched_count == 0:
            try:
                result = self.orders.update_one(
                    {"tracking_number": int(tracking)},
                    {"$set": {"status": "cancelled"}},
                )
            except (ValueError, TypeError):
                return False
        return result.modified_count > 0

    # ---- Users ----
    def get_user_by_id(self, user_id: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(user_id, str):
                return self.users.find_one({"_id": ObjectId(user_id)})
            return self.users.find_one({"_id": user_id})
        except (InvalidId, Exception):
            return None


mongodb = _MongoDB()
