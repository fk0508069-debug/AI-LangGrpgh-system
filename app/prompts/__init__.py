"""Prompt registry."""

from app.prompts.clarification import CLARIFICATION_PROMPT
from app.prompts.product_answer import PRODUCT_ANSWER_PROMPT
from app.prompts.query_understanding import QUERY_UNDERSTANDING_PROMPT
from app.prompts.rag import RAG_PROMPT

PROMPT_VERSIONS = {
    "query_understanding": "1.0",
    "clarification": "1.0",
    "product_answer": "1.0",
    "rag": "1.0",
}

__all__ = [
    "QUERY_UNDERSTANDING_PROMPT",
    "CLARIFICATION_PROMPT",
    "PRODUCT_ANSWER_PROMPT",
    "RAG_PROMPT",
    "PROMPT_VERSIONS",
]