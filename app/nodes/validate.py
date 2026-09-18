# app/nodes/validate.py
import logging
from app.state import GraphState
from app.services.llm import get_llm
from app.prompts.validation_prompt import VALIDATION_PROMPT

logger = logging.getLogger(__name__)


def validate_node(state: GraphState) -> dict:
    context = state.get("context", "")
    answer = state.get("answer", "").strip()
    question = state.get("question", "")

    logger.info("Validating answer against context...")

    if not answer:
        logger.warning("Validation rejected: empty answer.")
        return {"validation_result": False}

    lower_answer = answer.lower()
    refusal_phrases = [
        "could not find",
        "couldn't find",
        "do not have",
        "don't have",
        "no information",
        "not in the company",
        "unable to find",
        "cannot find",
    ]
    is_refusal = any(p in lower_answer for p in refusal_phrases)

    # A refusal when the context is empty is CORRECT behavior - accept it.
    if is_refusal and not context.strip():
        logger.info("Validation accepted: correct refusal with empty context.")
        return {"validation_result": True}

    llm = get_llm()
    chain = VALIDATION_PROMPT | llm

    try:
        response = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer,
        })
        result = response.content.strip().upper()

        if "INVALID" in result:
            is_valid = False
        elif "VALID" in result:
            is_valid = True
        else:
            logger.warning(f"Unexpected validation response: {result!r}")
            is_valid = False

        logger.info(f"Validation result: {'VALID' if is_valid else 'INVALID'}")
        return {"validation_result": is_valid}
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"validation_result": False}