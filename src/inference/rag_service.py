from __future__ import annotations

import os
from pathlib import Path

from src.data.ingestion import DocumentIngestionPipeline
from src.data.schemas import Answer
from src.inference.model_manager import ModelManager
from src.inference.predictor import Predictor
from src.models.text_model import TextModel
from src.rag.answer_strategy import AnswerStrategy
from src.rag.pipeline import RAGPipeline
from src.rag.qdrant_retriever import QdrantRetriever
from src.rag.reranker import CrossEncoderReranker


class RAGService:

    """
    Service RAG complet d'AFRICA-LMM.

    Ingestion:
        Document → Chunks → Qdrant

    Question:
        Question → Retrieval → Reranking → Answer Strategy
        → Answer + citations
    """

    def __init__(
        self,
        model_name: str,
        embedding_model: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        collection_name: str = "africa_lmm_documents",
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
    ) -> None:
        qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
        self.retriever = QdrantRetriever(
            embedding_model=embedding_model,
            collection_name=collection_name,
            host=qdrant_host,
            port=qdrant_port,
        )

        self.model = TextModel(
            model_name=model_name,
        )

        self.model_manager = ModelManager(self.model)

        self.reranker = CrossEncoderReranker()

        self.answer_strategy = AnswerStrategy(
            model=self.model,
            reranker_model=self.reranker,
        )

        self.rag_pipeline = RAGPipeline(
            retriever=self.retriever,
            model=self.model,
            reranker=self.reranker,
            answer_strategy=self.answer_strategy,
        )

        self.predictor = Predictor(
            pipeline=self.rag_pipeline,
        )

        self.ingestion = DocumentIngestionPipeline(
            retriever=self.retriever,
        )

    def load_model(self) -> None:
        """Load the language model."""
        self.model_manager.load()

    def unload_model(self) -> None:
        """Unload the language model."""
        self.model_manager.unload()

    def ingest_document(self, path: str | Path) -> None:
        """Ingest a document into Qdrant."""
        self.ingestion.ingest(path)

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        conversation_context: str | None = None,
    ) -> Answer:
        """Ask a question using the indexed documents."""
        self.model_manager.load()

        return self.predictor.predict(
            question=question,
            top_k=top_k,
            document_ids=document_ids,
            conversation_context=conversation_context,
        )
