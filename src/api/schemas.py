from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class CitationResponse(BaseModel):
    document_id: str
    page_number: int | None = None
    chunk_id: str | None = None


class RetrievedChunkResponse(BaseModel):
    chunk: dict[str, Any]
    score: float
    rank: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    conversation_id: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    pages: int
    size_bytes: int
    indexed: bool


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int
class ConversationCreateRequest(BaseModel):
    title: str = "Nouvelle conversation"


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageResponse]


class ConversationsResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int
