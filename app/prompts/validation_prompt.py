# app/prompts/validation_prompt.py
from langchain_core.prompts import ChatPromptTemplate

VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Keep answers under 80 words unless the user asks for details. Do not use numbered lists unless necessary. Give only the next action."
        "You are a strict fact-checker. Your job is to verify if the 'Answer' "
        "is fully supported by the 'Context'. "
        "If the answer contains information NOT found in the context, or if it "
        "invents a policy, output 'INVALID'. "
        "If the answer is fully supported, output 'VALID'. "
        "Output ONLY the word 'VALID' or 'INVALID'."
    )),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])