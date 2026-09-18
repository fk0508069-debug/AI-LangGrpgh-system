# app/services/vector_store.py
import logging
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import DB_PATH, EMBEDDING_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)

_embeddings = None
_vector_db = None


def get_vector_store() -> Chroma:
    global _vector_db, _embeddings
    if _vector_db is None:
        logger.info("Initializing ChromaDB connection...")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=_embeddings,
        )
    return _vector_db


def retrieve_documents(
    query: str,
    category: Optional[str] = None,
    k: int = RETRIEVAL_K,
) -> List[Document]:
    vector_db = get_vector_store()

    search_kwargs = {"k": k}
    if category and category != "general":
        search_kwargs["filter"] = {"category": category}
        logger.info(f"Retrieving top {k} docs for '{query}' filter={category!r}")
    else:
        logger.info(f"Retrieving top {k} docs for '{query}' (no filter)")

    try:
        # MMR gives diverse chunks instead of 5 near-duplicates
        results = vector_db.max_marginal_relevance_search(
            query,
            fetch_k=max(k * 3, 15),
            lambda_mult=0.5,
            **search_kwargs,
        )
        return results
    except Exception as e:
        logger.error(f"Error during vector retrieval: {e}")
        return []