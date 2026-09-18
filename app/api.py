# app/api.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from app.graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Base Chatbot API")

# Initialize the graph once at startup
try:
    graph = build_graph()
    logger.info("Graph loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load graph: {e}")
    graph = None

# In-memory session store
session_store: Dict[str, List] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str] = []


@app.get("/")
async def root():
    return {"status": "Knowledge Base Chatbot API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "graph_loaded": graph is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized. Check server logs.")

    logger.info(f"[{request.session_id}] User: {request.message}")

    history = session_store.get(request.session_id, [])

    initial_state = {
        "original_question": request.message,
        "question": "",
        "chat_history": history,
        "query_type": "",
        "documents": [],
        "context": "",
        "answer": "",
        "validation_result": False,
        "feedback": {},
    }

    try:
        result = graph.invoke(initial_state)
        session_store[request.session_id] = result["chat_history"]

        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in result.get("documents", [])
        ]))

        return ChatResponse(
            session_id=request.session_id,
            answer=result["answer"],
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[{request.session_id}] Graph error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    session_store.pop(session_id, None)
    return {"status": "session cleared", "session_id": session_id}