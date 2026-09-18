# app/prompts/classify_prompt.py
from langchain_core.prompts import ChatPromptTemplate

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query classifier for a knowledge base. "
        "Classify the user's question into exactly ONE of the following categories:\n"
        "- 'general': Greetings, chit-chat, or questions not related to the knowledge base.\n"
        "- 'docs': Questions about documentation, manuals, or guides.\n"
        "- 'faq': Frequently asked questions or general help topics.\n"
        "- 'technical': Questions about technical specifications or processes.\n"
        "- 'unknown': If the category cannot be determined.\n\n"
        "Output ONLY the category name. Do not include punctuation or explanation."
    )),
    ("human", "{question}")
])