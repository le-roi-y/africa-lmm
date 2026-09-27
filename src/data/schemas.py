from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    language: str | None = None
    country: str | None = None
    domain: str | None = None
    year: int | None = None
    source: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentPage(BaseModel):
    page_number: int
    text: str = ""
    images: list[str] = Field(default_factory=list)
    tables: list[Any] = Field(default_factory=list)


class Document(BaseModel):
    metadata: DocumentMetadata
    text: str = ""
    pages: list[DocumentPage] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


class Citation(BaseModel):
    document_id: str
    page_number: int | None = None
    chunk_id: str | None = None


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
