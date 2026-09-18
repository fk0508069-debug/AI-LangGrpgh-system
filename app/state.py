# app/state.py
from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import operator

class GraphState(TypedDict):
    """
    Shared state object passed between LangGraph nodes.
    """
    # Input
    original_question: str  # <-- ADDED: Keep the raw user input
    question: str           # <-- UPDATED: Will hold the corrected query
    
    # Conversation Memory
    chat_history: Annotated[List[BaseMessage], operator.add]
    
    # Classification
    query_type: str  
    
    # Retrieval
    documents: List[Document]
    context: str
    
    # Generation
    answer: str
    
    # Validation
    validation_result: bool
    
    # Feedback
    feedback: dict