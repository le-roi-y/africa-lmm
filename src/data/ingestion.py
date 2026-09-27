from __future__ import annotations

from pathlib import Path

from src.data.document_engine import DocumentEngine
from src.data.schemas import Document
from src.data.splitter import TextSplitter
from src.data.validator import DocumentValidator
from src.rag.retriever import Retriever


class DocumentIngestionPipeline:
    """
    Pipeline d'ingestion des documents pour AFRICA-LMM.

    PDF
        ↓
    Extraction du texte
        ↓
    Validation
        ↓
    Chunking
        ↓
    Indexation vectorielle
    """

    def __init__(
        self,
        retriever: Retriever,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.document_engine = DocumentEngine()
        self.validator = DocumentValidator()
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = retriever

    def ingest(self, path: str | Path) -> Document:
        """Load, validate, split and index a PDF document."""
        document = self.document_engine.load_pdf(path)

        self.validator.validate(document)

        chunks = self.splitter.split(document)

        self.retriever.index(chunks)

        return document
