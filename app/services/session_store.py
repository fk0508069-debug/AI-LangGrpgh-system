"""In-memory session registry."""

from __future__ import annotations

import threading
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import get_settings
from app.state import SessionData


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData()
            return self._sessions[session_id]

    def save_turn(self, session_id: str, question: str, answer: str) -> None:
        s = get_settings()
        with self._lock:
            sess = self.get(session_id)
            sess.history.extend(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )
            cap = max(2, s.max_history_messages)
            sess.history = sess.history[-cap:]

    def history_as_dicts(self, session_id: str) -> List[dict]:
        sess = self.get(session_id)
        out: List[dict] = []
        for m in sess.history:
            role = "assistant" if isinstance(m, AIMessage) else "user"
            out.append({"role": role, "content": str(m.content)})
        return out

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()


session_store = SessionStore()