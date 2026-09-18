# app/nodes/generate.py
import logging
from langchain_core.messages import AIMessage
from app.state import GraphState
from app.services.llm import get_llm
from app.services.chat_history import format_chat_history
from app.prompts.rag_prompt import RAG_PROMPT

logger = logging.getLogger(__name__)

# Any past AI answer containing one of these fragments gets dropped
# before being sent back to the LLM, so refusals don't snowball.
_REFUSAL_FRAGMENTS = (
    "could not find",
    "couldn't find",
    "could not be fully verified",
    "no relevant",
    "no information",
    "not in the knowledge base",
    "not in the company documents",
)


def _filter_poisoned_history(messages):
    """Drop past AI refusals so the model doesn't parrot them."""
    cleaned = []
    for m in messages:
        if isinstance(m, AIMessage):
            content = (m.content or "").lower()
            if any(frag in content for frag in _REFUSAL_FRAGMENTS):
                continue
        cleaned.append(m)
    return cleaned


def generate_node(state: GraphState) -> dict:
    question = state["question"]
    context = state.get("context", "")
    history = state.get("chat_history", [])

    logger.info("Generating answer...")

    history = _filter_poisoned_history(history)

    if not context.strip():
        logger.warning("No context - LLM will produce a refusal.")
        context = "(No relevant company documents were found for this question.)"

    llm = get_llm()
    chain = RAG_PROMPT | llm
    formatted_history = format_chat_history(history)

    try:
        response = chain.invoke({
            "context": context,
            "chat_history": formatted_history,
            "question": question,
        })
        answer = response.content.strip()
        logger.info("Answer generated successfully.")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"answer": "I encountered an error while trying to generate an answer."}