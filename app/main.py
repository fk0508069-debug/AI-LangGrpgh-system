"""CLI runner — for interactive testing without FastAPI."""

from __future__ import annotations

from app.logging_config import setup_logging

setup_logging()

import logging

from app.graph import build_graph
from app.services.session_store import session_store

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting E-Commerce AI Assistant (CLI)...")
    graph = build_graph()

    session_id = "cli-session"
    print("Chatbot ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        state = {
            "session_id": session_id,
            "original_question": user_input,
            "chat_history": [],
            "documents": [],
            "context": "",
            "answer": "",
            "validation_result": False,
        }

        try:
            result = graph.invoke(state)
            print(f"\nAssistant: {result.get('answer', '')}\n")
        except Exception as e:
            logger.exception("Graph error")
            print(f"\n[error] {e}\n")


if __name__ == "__main__":
    main()