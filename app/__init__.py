"""E-Commerce AI Assistant — LangGraph edition."""

from __future__ import annotations

import os

__version__ = "1.0.0"
__version_info__ = tuple(int(p) for p in __version__.split("."))

if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_PROJECT", "ecommerce-langgraph")

__all__ = ["__version__", "__version_info__", "get_settings", "setup_logging", "build_graph"]


def __getattr__(name: str):
    if name == "get_settings":
        from app.config import get_settings
        return get_settings
    if name == "setup_logging":
        from app.logging_config import setup_logging
        return setup_logging
    if name == "build_graph":
        from app.graph import build_graph
        return build_graph
    raise AttributeError(f"module 'app' has no attribute {name!r}")