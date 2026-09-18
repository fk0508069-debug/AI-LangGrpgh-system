# app/nodes/memory.py
import logging
from langchain_core.messages import HumanMessage, AIMessage
from app.state import GraphState

logger = logging.getLogger(__name__)

_REFUSAL_FRAGMENTS = (
    "could not find",
    "couldn't find",
    "could not be fully verified",
    "not in the knowledge base",
    "not in the company documents",
    "no relevant",
)


def _is_refusal(text: str) -> bool:
    lower = (text or "").lower()
    return any(frag in lower for frag in _REFUSAL_FRAGMENTS)


def update_memory_node(state: GraphState) -> dict:
    """
    Append the user question and AI answer to chat history.
    Refusals are NOT persisted — otherwise the LLM will keep parroting them.
    """
    question = state["original_question"]
    answer = state.get("answer", "")

    if _is_refusal(answer):
        logger.info("Skipping memory write: answer is a refusal.")
        # Still return the channel so LangGraph doesn't complain, but
        # with an empty list to add nothing.
        return {"chat_history": []}

    new_messages = [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]

    logger.info("Updating chat history.")
    return {"chat_history": new_messages}