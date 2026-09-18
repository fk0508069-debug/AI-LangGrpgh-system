"""FastAPI entry point."""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.logging_config import setup_logging

setup_logging()

from app.config import get_settings
from app.graph import build_graph
from app.health import build_health_report
from app.schemas import ClearResponse, HistoryMessage, QueryRequest, QueryResponse
from app.services.session_store import session_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(x_api_key: Optional[str] = Security(api_key_header)) -> None:
    expected = get_settings().fastapi_api_key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="E-Commerce AI Assistant (LangGraph)")

try:
    graph = build_graph()
    logger.info("Graph loaded successfully.")
except Exception as e:
    logger.error("Failed to build graph: %s", e, exc_info=True)
    graph = None


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "E-Commerce AI Assistant is running"}


@app.get("/health")
def health_check():
    report = build_health_report()
    report["graph_loaded"] = graph is not None
    return report


# ---------------------------------------------------------------------------
# Debug endpoints (no auth — local dev only)
# ---------------------------------------------------------------------------
@app.get("/debug/config")
def debug_config():
    """Show what config the app actually loaded from .env."""
    try:
        s = get_settings()
    except Exception as e:
        return {"error": f"Failed to load settings: {e}"}

    # Mask secrets — only show last 4 chars
    def mask(v: str) -> str:
        if not v:
            return "<empty>"
        return f"***{v[-4:]}" if len(v) > 4 else "***"

    return {
        "mongodb_uri": s.mongodb_uri,
        "mongodb_db": s.mongodb_db,
        "mongo_collection": s.mongo_collection,
        "llm_model": s.llm_model,
        "temperature": s.temperature,
        "embedding_model": s.embedding_model,
        "document_path": s.document_path,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "retrieval_k": s.retrieval_k,
        "max_history_messages": s.max_history_messages,
        "product_base_url": s.product_base_url,
        "groq_api_key": mask(s.groq_api_key),
        "fastapi_api_key": mask(s.fastapi_api_key),
    }


@app.get("/debug/db")
def debug_db():
    """Show what MongoDB sees — DBs, collections, doc count, sample fields."""
    from app.config import get_settings as _gs
    from app.services.mongodb import mongodb

    s = _gs()

    try:
        db = mongodb._ensure()
        client = mongodb._client
        databases = client.list_database_names()
    except Exception as e:
        return {"error": f"Cannot reach MongoDB: {e}", "uri": s.mongodb_uri}

    try:
        collections = db.list_collection_names()
    except Exception as e:
        collections = [f"<error listing collections: {e}>"]

    result: dict = {
        "uri": s.mongodb_uri,
        "db_name_used": s.mongodb_db,
        "collection_used": s.mongo_collection,
        "databases_on_server": databases,
        "collections_in_db": collections,
    }

    try:
        count = mongodb.products.estimated_document_count()
        result["product_doc_count"] = count

        sample = mongodb.products.find_one({}) or {}
        sample.pop("_id", None)
        result["sample_doc_fields"] = list(sample.keys())
        # also show a type preview of each field
        result["sample_doc_field_types"] = {
            k: type(v).__name__ for k, v in sample.items()
        }
    except Exception as e:
        result["error_querying_products"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Chat endpoints (auth required)
# ---------------------------------------------------------------------------
@app.post(
    "/ask",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
)
def ask_question(request: QueryRequest) -> QueryResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized.")

    session_id = request.session_id
    logger.info("[%s] User: %s", session_id, request.question)

    initial_state = {
        "session_id": session_id,
        "original_question": request.question,
        "chat_history": [],
        "documents": [],
        "context": "",
        "answer": "",
        "validation_result": False,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        logger.exception("[%s] Graph error", session_id)
        raise HTTPException(status_code=500, detail=str(e))

    history = [
        HistoryMessage(role=m["role"], content=m["content"])
        for m in session_store.history_as_dicts(session_id)
    ]
    return QueryResponse(
        answer=result.get("answer") or "",
        session_id=session_id,
        history=history,
    )


@app.post(
    "/clear",
    response_model=ClearResponse,
    dependencies=[Depends(require_api_key)],
)
def clear_session(request: QueryRequest) -> ClearResponse:
    session_store.clear(request.session_id)
    return ClearResponse(status="cleared", session_id=request.session_id)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    session_store.clear(session_id)
    return {"status": "session cleared", "session_id": session_id}


@app.get("/debug/rag")
def debug_rag(q: str = "fighting in the company"):
    """Show exactly what the RAG retriever returns for a query."""
    from app.services.vector_store import get_retriever, get_vectorstore
    from app.config import get_settings
    import os

    s = get_settings()

    result: dict = {
        "document_path_config": s.document_path,
        "document_path_exists": os.path.exists(s.document_path),
    }

    # Try alternate paths
    base = os.path.splitext(s.document_path)[0]
    for cand in [s.document_path, base + ".txt", "data/company_policy.txt",
                 "data/policies/company_policy.txt"]:
        result.setdefault("file_checks", {})[cand] = os.path.exists(cand)

    try:
        vs = get_vectorstore()
        result["index_size"] = vs.index.ntotal
    except Exception as e:
        result["index_error"] = str(e)
        return result

    try:
        retriever = get_retriever()
        docs = retriever.invoke(q)
        result["query"] = q
        result["num_docs_returned"] = len(docs)
        result["docs"] = [
            {
                "source": (d.metadata or {}).get("source", "?"),
                "length": len(d.page_content or ""),
                "preview": (d.page_content or "")[:300],
            }
            for d in docs
        ]
    except Exception as e:
        result["retrieval_error"] = str(e)

    return result