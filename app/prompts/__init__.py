"""
Company Policy RAG Chatbot.

A LangGraph-powered retrieval-augmented generation pipeline for
querying company policy, HR, and product documents.

Public surface (Tier 2):
    from app import __version__, get_settings, setup_logging
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
# Single source of truth. Bump here; /health and logs pick it up automatically.
__version__ = "0.2.0"
__version_info__ = tuple(int(p) for p in __version__.split("."))

# ---------------------------------------------------------------------------
# Observability bootstrap
# ---------------------------------------------------------------------------
# If LangSmith tracing env vars are present, tag the project name without
# forcing the langsmith dependency. Runs exactly once at import time.
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_PROJECT", "company-policy-rag")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "__version__",
    "__version_info__",
    "get_settings",
    "setup_logging",
]


# ---------------------------------------------------------------------------
# Lazy loading (PEP 562)
# ---------------------------------------------------------------------------
# Heavy imports (graph, chromadb, HF embeddings) must NOT run on `import app`.
# Only resolve them when the attribute is actually accessed.
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


def __dir__():
    return sorted(list(globals().keys()) + __all__)