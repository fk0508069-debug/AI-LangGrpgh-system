"""FastAPI request/response schemas."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100)


class HistoryMessage(BaseModel):
    role: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    history: List[HistoryMessage]


class ClearResponse(BaseModel):
    status: str
    session_id: str