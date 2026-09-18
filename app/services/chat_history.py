# app/services/chat_history.py
import logging
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 6  # Keep last 3 exchanges (Human + AI)

def format_chat_history(history: List[BaseMessage]) -> str:
    """Formats the chat history into a readable string for the prompt."""
    if not history:
        return "No previous conversation."
    
    # Trim to the last N messages to save tokens and prevent context overflow
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    
    formatted = []
    for msg in recent_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted)

def summarize_history_if_needed(history: List[BaseMessage], llm) -> List[BaseMessage]:
    """
    If the history gets too long, we could summarize it here.
    For now, we rely on the trimming in format_chat_history.
    """
    # Placeholder for future summarization logic
    return history