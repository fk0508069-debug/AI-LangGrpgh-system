# ingest.py
import os
import logging
from typing import List
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    POLICIES_DIR, HR_DIR, PRODUCTS_DIR, DB_PATH,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_documents_from_dir(directory: str, category: str) -> List[Document]:
    """Loads all .docx and .pdf files from a directory and tags them with a category."""
    documents = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return documents

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            continue
        
        try:
            docs = loader.load()
            for doc in docs:
                # Requirement 5: Metadata
                doc.metadata["source"] = filename
                doc.metadata["category"] = category
                doc.metadata["document_type"] = filename.split(".")[-1]
                # Page is automatically added by PyPDFLoader, Docx2txt doesn't have pages, so we set 0
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {filename}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            
    return documents

def ingest():
    logger.info("Starting ingestion pipeline...")
    
    # 1. Load documents with metadata
    policy_docs = load_documents_from_dir(POLICIES_DIR, "policy")
    hr_docs = load_documents_from_dir(HR_DIR, "hr")
    product_docs = load_documents_from_dir(PRODUCTS_DIR, "product")
    
    all_documents = policy_docs + hr_docs + product_docs
    if not all_documents:
        logger.error("No documents found to ingest. Exiting.")
        return

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_documents)
    logger.info(f"Split into {len(chunks)} chunks.")

    # 3. Embed and store
    
    
    logger.info("Storing in ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB at {DB_PATH}.")

if __name__ == "__main__":
    ingest()