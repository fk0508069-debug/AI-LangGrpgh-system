"""Health/version payload for /health."""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import app
from app.config import get_settings


def build_health_report() -> dict:
    s = get_settings()
    return {
        "status": "healthy",
        "app_version": app.__version__,
        "python_version": platform.python_version(),
        "llm_model": s.llm_model,
        "embedding_model": s.embedding_model,
        "retrieval_k": s.retrieval_k,
        "db_name": s.mongodb_db,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }