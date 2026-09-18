# app/graph.py
import logging
from langgraph.graph import StateGraph, START, END
from app.state import GraphState
from app.nodes.rewrite import rewrite_query_node
from app.nodes.classify import classify_node
from app.nodes.retrieve import retrieve_node
from app.nodes.generate import generate_node
from app.nodes.validate import validate_node
from app.nodes.memory import update_memory_node
from app.nodes.fallback import fallback_node

logger = logging.getLogger(__name__)

def build_graph():
    """Builds and compiles the LangGraph state machine."""
    workflow = StateGraph(GraphState)
    
    # Add Nodes
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("memory", update_memory_node)
    workflow.add_node("fallback", fallback_node)
    
    # Define Edges
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    
    # Conditional Edge for Validation (Requirement 12)
    workflow.add_conditional_edges(
        "validate",
        lambda state: state["validation_result"],
        {
            True: "memory",       # If valid, update memory and finish
            False: "fallback"     # If invalid, route to fallback
        }
    )
    
    workflow.add_edge("fallback", "memory")
    workflow.add_edge("memory", END)
    
    logger.info("LangGraph compiled successfully.")
    return workflow.compile()