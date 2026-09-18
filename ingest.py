"""One-shot ingestion: load policy files → chunk → embed → save FAISS index.

Run from project root:
    python -m app.ingest
"""

from __future__ import annotations

import logging
import os
import sys

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("ingest")

# Where to persist the FAISS index
INDEX_DIR = "data/faiss_index"

# Which files to ingest — add/remove as needed
SOURCE_FILES = [
    "data/company_policy.txt",
    "data/policies/company_policy.pdf",
    # "data/policies/company_policy.docx",  # skip if broken
]


def load_file(path: str):
    if not os.path.exists(path):
        logger.warning("Skipping missing file: %s", path)
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt" or ext == ".md":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    else:
        logger.warning("Unsupported file type: %s", path)
        return []
    try:
        docs = loader.load()
        logger.info("Loaded %s → %d doc(s)", path, len(docs))
        return docs
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return []


def main() -> None:
    s = get_settings()
    logger.info("Ingestion starting.")
    logger.info("Chunk size=%d overlap=%d", s.chunk_size, s.chunk_overlap)

    # 1. Load
    all_docs = []
    for path in SOURCE_FILES:
        all_docs.extend(load_file(path))

    if not all_docs:
        logger.error("No documents loaded. Nothing to ingest.")
        sys.exit(1)

    logger.info("Total loaded: %d documents", len(all_docs))

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)
    logger.info("Produced %d chunks", len(chunks))

    # 3. Embed
    logger.info("Embedding with %s ...", s.embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=s.embedding_model)

    # 4. Build + persist
    logger.info("Building FAISS index...")
    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    logger.info("Saved index to %s (%d vectors)", INDEX_DIR, vs.index.ntotal)


if __name__ == "__main__":
    main()