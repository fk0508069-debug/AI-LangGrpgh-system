"""Central configuration. Reads from .env via pydantic."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Real process env wins over .env (safer in Docker/K8s).
load_dotenv(override=False)

T = TypeVar("T")

_DEFAULTS: Dict[str, Any] = {
    "mongodb_db": "ecommerce",
    "mongo_collection": "products",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "openai/gpt-oss-120b",
    "temperature": 0.0,
    "document_path": "data/company_policy.txt",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "retrieval_k": 5,
    "max_history_messages": 20,
    "product_base_url": "https://demo-fiver-project.vercel.app/products",
}


def _first_env(*names: str) -> Optional[str]:
    for key in names:
        v = os.getenv(key)
        if v is not None:
            v = v.strip()
            if v:
                return v
    return None


def _coerce(raw: Optional[str], cast: Callable[[str], T], default: T, name: str) -> T:
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid "
            f"{cast.__name__}: {exc}"
        ) from exc


def _get_str(name: str, *aliases: str, default: str = "") -> str:
    return _first_env(name, *aliases) or default


def _get_typed(name: str, *aliases: str, cast: Callable[[str], T], default: T) -> T:
    return _coerce(_first_env(name, *aliases), cast, default, name)


def _extract_db_from_uri(uri: str) -> str:
    """Parse 'mongodb://host:port/dbname?opts' → 'dbname'."""
    try:
        after_host = uri.split("://", 1)[-1]
        path = after_host.split("/", 1)[1] if "/" in after_host else ""
        return path.split("?", 1)[0].strip()
    except Exception:
        return ""


class Settings(BaseModel):
    # Secrets
    groq_api_key: str = Field(..., min_length=10)
    fastapi_api_key: str = Field(..., min_length=4)

    # Mongo
    mongodb_uri: str = Field(..., min_length=5)
    mongodb_db: str = Field(_DEFAULTS["mongodb_db"], min_length=1)
    mongo_collection: str = Field(_DEFAULTS["mongo_collection"], min_length=1)

    # Models
    embedding_model: str = _DEFAULTS["embedding_model"]
    llm_model: str = _DEFAULTS["llm_model"]
    temperature: float = Field(_DEFAULTS["temperature"], ge=0.0, le=2.0)

    # RAG
    document_path: str = _DEFAULTS["document_path"]
    chunk_size: int = Field(_DEFAULTS["chunk_size"], gt=0)
    chunk_overlap: int = Field(_DEFAULTS["chunk_overlap"], ge=0)
    retrieval_k: int = Field(_DEFAULTS["retrieval_k"], gt=0)

    # Session
    max_history_messages: int = Field(_DEFAULTS["max_history_messages"], gt=0)

    # URLs
    product_base_url: str = _DEFAULTS["product_base_url"]

    @field_validator("product_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        groq_key = _first_env("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables.")

        fastapi_key = _first_env("FASTAPI_API_KEY")
        if not fastapi_key:
            raise ValueError("FASTAPI_API_KEY is missing from environment variables.")

        uri = _first_env("MONGODB_URI", "MONGO_URI")
        if not uri:
            raise ValueError("MONGODB_URI is missing from environment variables.")

        db_name = (
            _first_env("MONGO_DB", "MONGODB_DB")
            or _extract_db_from_uri(uri)
            or _DEFAULTS["mongodb_db"]
        )

        return cls(
            groq_api_key=groq_key,
            fastapi_api_key=fastapi_key,
            mongodb_uri=uri,
            mongodb_db=db_name,
            mongo_collection=_get_str(
                "MONGO_COLLECTION", "MONGODB_COLLECTION",
                default=_DEFAULTS["mongo_collection"],
            ),
            llm_model=_get_str("LLM_MODEL", default=_DEFAULTS["llm_model"]),
            temperature=_get_typed(
                "TEMPERATURE", "LLM_TEMPERATURE",
                cast=float, default=_DEFAULTS["temperature"],
            ),
            embedding_model=_get_str(
                "EMBEDDING_MODEL", default=_DEFAULTS["embedding_model"],
            ),
            document_path=_get_str(
                "DOCUMENT_PATH", default=_DEFAULTS["document_path"],
            ),
            chunk_size=_get_typed(
                "CHUNK_SIZE", cast=int, default=_DEFAULTS["chunk_size"],
            ),
            chunk_overlap=_get_typed(
                "CHUNK_OVERLAP", cast=int, default=_DEFAULTS["chunk_overlap"],
            ),
            retrieval_k=_get_typed(
                "RETRIEVAL_K", cast=int, default=_DEFAULTS["retrieval_k"],
            ),
            max_history_messages=_get_typed(
                "MAX_HISTORY_MESSAGES",
                cast=int, default=_DEFAULTS["max_history_messages"],
            ),
            product_base_url=_get_str(
                "PRODUCT_BASE_URL", default=_DEFAULTS["product_base_url"],
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]