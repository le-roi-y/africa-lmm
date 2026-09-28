# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.schemas import (
    CitationResponse,
    ConversationCreateRequest,
    ConversationResponse,
    ConversationsResponse,
    ConversationSummary,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
)
from src.database.session import get_db
from src.inference.conversation_service import ConversationService
from src.inference.rag_service import RAGService
from src.inference.title_generator import generate_conversation_title

APP_VERSION = "0.1.0"
UPLOAD_DIR = Path("data/uploads")


app = FastAPI(
    title="AFRICA-LMM API",
    description=(
        "Multimodal AI platform for African documents. "
        "Provides document ingestion, retrieval, and question answering."
    ),
    version=APP_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag_service = RAGService(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="africa-lmm",
        version=APP_VERSION,
    )


@app.post("/query", response_model=QueryResponse, tags=["inference"])
def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
) -> QueryResponse:
    conversation_service = ConversationService(db)

    if request.conversation_id:
        conversation = conversation_service.get(request.conversation_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        conversation_id = str(conversation.id)
    else:
        conversation = conversation_service.create(
            title=generate_conversation_title(request.question),
        )
        conversation_id = str(conversation.id)

    recent_messages = (
        conversation_service.recent_messages(
            conversation_id=conversation_id,
            limit=8,
        )
        if request.conversation_id
        else []
    )

    # Pour le retrieval, conserver les questions précédentes
    # permet de résoudre les suivis courts sans polluer
    # l'embedding avec les réponses générées par le LLM.
    previous_questions = [
        message.content.strip()
        for message in recent_messages
        if message.role == "user" and message.content.strip()
    ]

    conversation_context = "\n".join(previous_questions[-3:])

    conversation_service.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.question,
    )

    try:
        answer = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
            conversation_context=conversation_context or None,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    citations = [
        CitationResponse(
            document_id=citation.document_id,
            page_number=citation.page_number,
            chunk_id=citation.chunk_id,
        )
        for citation in answer.citations
    ]

    conversation_service.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer.answer,
        citations=[citation.model_dump() for citation in citations],
    )

    return QueryResponse(
        question=answer.question,
        answer=answer.answer,
        conversation_id=conversation_id,
        citations=citations,
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk=result.chunk.model_dump(),
                score=result.score,
                rank=result.rank,
            )
            for result in answer.retrieved_chunks
        ],
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents() -> DocumentsResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    documents: list[DocumentInfo] = []

    for path in sorted(UPLOAD_DIR.glob("*.pdf")):
        try:
            with fitz.open(str(path)) as pdf:
                pages = len(pdf)
        except Exception:
            pages = 0

        documents.append(
            DocumentInfo(
                document_id=path.stem,
                filename=path.name,
                pages=pages,
                size_bytes=path.stat().st_size,
                indexed=True,
            )
        )

    return DocumentsResponse(
        documents=documents,
        total=len(documents),
    )


@app.post(
    "/conversations",
    response_model=ConversationResponse,
    tags=["conversations"],
)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
) -> ConversationResponse:
    service = ConversationService(db)
    conversation = service.create(title=request.title)

    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        messages=[],
    )


@app.get(
    "/conversations",
    response_model=ConversationsResponse,
    tags=["conversations"],
)
def list_conversations(
    db: Session = Depends(get_db),
) -> ConversationsResponse:
    service = ConversationService(db)
    conversations = service.list()

    return ConversationsResponse(
        conversations=[
            ConversationSummary(
                id=str(conversation.id),
                title=conversation.title,
                created_at=conversation.created_at.isoformat(),
                updated_at=conversation.updated_at.isoformat(),
            )
            for conversation in conversations
        ],
        total=len(conversations),
    )


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["conversations"],
)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> ConversationResponse:
    service = ConversationService(db)
    conversation = service.get(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        messages=[
            MessageResponse(
                id=str(message.id),
                role=message.role,
                content=message.content,
                citations=message.citations,
                created_at=message.created_at.isoformat(),
            )
            for message in conversation.messages
        ],
    )


@app.delete(
    "/conversations/{conversation_id}",
    tags=["conversations"],
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    service = ConversationService(db)

    if not service.delete(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {"deleted": True}


@app.post(
    "/documents",
    response_model=IngestResponse,
    tags=["documents"],
)
async def ingest_document(
    file: UploadFile = File(...),
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination = UPLOAD_DIR / Path(file.filename).name

    try:
        content = await file.read()
        destination.write_bytes(content)

        document = rag_service.ingestion.ingest(destination)

        return IngestResponse(
            document_id=document.metadata.document_id,
            filename=document.metadata.filename,
            pages=len(document.pages),
            message="Document ingested and indexed successfully.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@app.get(
    "/documents/{document_id}/file",
    tags=["documents"],
)
def get_document_file(document_id: str) -> FileResponse:
    if Path(document_id).name != document_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid document identifier.",
        )

    document_path = UPLOAD_DIR / f"{document_id}.pdf"

    if not document_path.exists() or not document_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return FileResponse(
        path=document_path,
        media_type="application/pdf",
        filename=document_path.name,
    )
