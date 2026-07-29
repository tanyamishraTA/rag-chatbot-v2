from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    model: Literal["gemini", "ollama"] = "ollama"


class Source(BaseModel):
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]