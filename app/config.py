"""Central configuration. Reads from .env via pydantic."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(override=True)


def _get(name: str, *aliases: str, default: str = "") -> str:
    """Return the first non-empty env var from name or any alias."""
    for key in (name, *aliases):
        v = os.getenv(key, "").strip()
        if v:
            return v
    return default


def _extract_db_from_uri(uri: str) -> str:
    """Parse 'mongodb://host:port/dbname?opts' → 'dbname'."""
    try:
        after_host = uri.split("://", 1)[-1]
        path = after_host.split("/", 1)[1] if "/" in after_host else ""
        return path.split("?", 1)[0].strip() or ""
    except Exception:
        return ""


class Settings(BaseModel):
    # Secrets
    groq_api_key: str = Field(..., min_length=10)
    fastapi_api_key: str = Field(..., min_length=4)

    # Mongo
    mongodb_uri: str = Field(..., min_length=5)
    mongodb_db: str = "ecommerce"
    mongo_collection: str = "products"

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0

    # RAG
    document_path: str = "data/company_policy.txt"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_k: int = 5

    # Session
    max_history_messages: int = 20

    # URLs
    product_base_url: str = "http://localhost:3000/details"

    @classmethod
    def from_env(cls) -> "Settings":
        groq_key = _get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables.")

        fastapi_key = _get("FASTAPI_API_KEY")
        if not fastapi_key:
            raise ValueError("FASTAPI_API_KEY is missing from environment variables.")

        uri = _get("MONGODB_URI", "MONGO_URI")
        if not uri:
            raise ValueError("MONGODB_URI is missing from environment variables.")

        # Resolve DB name: env var wins, else from URI, else default
        db_name = _get("MONGO_DB", "MONGODB_DB")
        if not db_name:
            db_name = _extract_db_from_uri(uri) or "ecommerce"

        return cls(
            groq_api_key=groq_key,
            fastapi_api_key=fastapi_key,
            mongodb_uri=uri,
            mongodb_db=db_name,
            mongo_collection=_get(
                "MONGO_COLLECTION", "MONGODB_COLLECTION", default="products"
            ),
            llm_model=_get(
                "LLM_MODEL", default="llama-3.3-70b-versatile"
            ),
            temperature=float(
                _get("TEMPERATURE", "LLM_TEMPERATURE", default="0.0")
            ),
            embedding_model=_get(
                "EMBEDDING_MODEL",
                default="sentence-transformers/all-MiniLM-L6-v2",
            ),
            document_path=_get(
                "DOCUMENT_PATH", default="data/company_policy.txt"
            ),
            chunk_size=int(_get("CHUNK_SIZE", default="500")),
            chunk_overlap=int(_get("CHUNK_OVERLAP", default="100")),
            retrieval_k=int(_get("RETRIEVAL_K", default="5")),
            max_history_messages=int(
                _get("MAX_HISTORY_MESSAGES", default="20")
            ),
            product_base_url=_get(
                "PRODUCT_BASE_URL",
                default="http://localhost:3000/details",
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


settings = get_settings()