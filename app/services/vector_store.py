"""FAISS vector store for RAG fallback."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_documents(path: str):
    """Load the RAG source with the loader matching its real type."""
    if not os.path.exists(path):
        logger.warning("RAG file not found: %s", path)
        return []

    ext = os.path.splitext(path)[1].lower()

    # docx
    if ext == ".docx":
        try:
            loader = Docx2txtLoader(path)
            docs = loader.load()
            if docs and any(d.page_content.strip() for d in docs):
                logger.info("Loaded .docx with %d document(s)", len(docs))
                return docs
            logger.warning(".docx loaded but yielded empty content.")
        except Exception as e:
            logger.error("Failed to load .docx (%s). Falling back to text loader.", e)
        # try as text (some people rename .txt → .docx)
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            if docs:
                logger.info("Loaded as plain text (fallback).")
                return docs
        except Exception as e:
            logger.error("Text fallback also failed: %s", e)
            return []

    # txt / md / anything else
    try:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return []


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    s = get_settings()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Resolve the path: prefer .txt over .docx if both exist
    path = s.document_path
    base = os.path.splitext(path)[0]
    candidates = [
        path,
        base + ".txt",
        "data/company_policy.txt",
        "data/policies/company_policy.txt",
    ]

    documents = []
    used_path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            loaded = _load_documents(candidate)
            if loaded and any(d.page_content.strip() for d in loaded):
                documents = loaded
                used_path = candidate
                break

    if not documents:
        logger.error("No valid RAG source document found. Using empty store.")
        return FAISS.from_texts(["(no documents loaded)"], embeddings)

    logger.info("Using RAG source: %s", used_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    # Cap to keep prompts small for Groq
    chunks = chunks[:10]
    logger.info("Building FAISS index over %d chunks...", len(chunks))
    return FAISS.from_documents(chunks, embeddings)


def get_retriever():
    s = get_settings()
    # Small k to keep context small
    k = min(s.retrieval_k, 3)
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
