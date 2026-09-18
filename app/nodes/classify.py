# app/nodes/classify.py
import logging
from app.state import GraphState
from app.services.llm import get_llm
from app.prompts.classify_prompt import CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

def classify_node(state: GraphState) -> dict:
    question = state["question"]
    logger.info(f"Classifying query: {question}")
    
    llm = get_llm()
    chain = CLASSIFY_PROMPT | llm
    
    try:
        response = chain.invoke({"question": question})
        query_type = response.content.strip().lower()
        
        # Validate the output
        valid_types = ["general", "policy", "hr", "product", "unknown"]
        if query_type not in valid_types:
            logger.warning(f"LLM returned invalid type: {query_type}. Defaulting to 'unknown'.")
            query_type = "unknown"
            
        logger.info(f"Query classified as: {query_type}")
        return {"query_type": query_type}
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {"query_type": "unknown"}