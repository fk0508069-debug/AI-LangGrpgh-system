# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = "data"
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
HR_DIR = os.path.join(DATA_DIR, "hr")
PRODUCTS_DIR = os.path.join(DATA_DIR, "products")
DB_PATH = "chroma_db"

# --- Models ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "groq/compound-mini"  # Groq model
LLM_TEMPERATURE = 0.0

# --- RAG Settings ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from environment variables.")
