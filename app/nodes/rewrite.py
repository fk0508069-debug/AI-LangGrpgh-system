# app/nodes/rewrite.py
import logging
from app.state import GraphState
from app.services.llm import get_llm
from app.prompts.rewrite_prompt import REWRITE_PROMPT

logger = logging.getLogger(__name__)

def rewrite_query_node(state: GraphState) -> dict:
    """
    Node to correct spelling and grammar in the user's question.
    """
    original_q = state["original_question"]
    logger.info(f"Rewriting query: '{original_q}'")
    
    llm = get_llm()
    chain = REWRITE_PROMPT | llm
    
    try:
        response = chain.invoke({"question": original_q})
        corrected_q = response.content.strip()
        
        # Fallback if the LLM returns an empty string
        if not corrected_q:
            corrected_q = original_q
            
        logger.info(f"Corrected query: '{corrected_q}'")
        
        # Update the state with the corrected question, 
        # but keep the original for feedback/evaluation later.
        return {
            "question": corrected_q,
            "original_question": original_q
        }
    except Exception as e:
        logger.error(f"Error during query rewriting: {e}")
        # If rewriting fails, proceed with the original question to avoid breaking the app
        return {"question": original_q, "original_question": original_q}