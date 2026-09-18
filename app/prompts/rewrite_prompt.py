# app/prompts/rewrite_prompt.py
from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict spelling correction assistant. "
        "Correct ONLY obvious spelling mistakes (e.g., 'femail' -> 'female', 'herassment' -> 'harassment'). "
        "DO NOT change the meaning, intent, or add new words. "
        "If the question has no typos, return it exactly as is. "
        "Output ONLY the corrected question. No explanations."
    )),
    ("human", "{question}")
])