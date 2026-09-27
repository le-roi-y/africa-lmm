from __future__ import annotations

from dataclasses import dataclass

from src.data.schemas import Answer, Document
from src.data.splitter import TextSplitter
from src.models.base import BaseModel
from src.rag.pipeline import RAGPipeline


@dataclass
class InferenceConfig:
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200


class InferencePipeline:
    """
    Pipeline principal d'inférence d'AFRICA-LMM.

    Document
        ↓
    Chunking
        ↓
    Indexation
        ↓
    Retrieval
        ↓
    Generation
        ↓
    Answer + citations
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        model: BaseModel,
        config: InferenceConfig | None = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.model = model
        self.config = config or InferenceConfig()

        self.splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def index_document(self, document: Document) -> None:
        chunks = self.splitter.split(document)

        self.rag_pipeline.retriever.index(chunks)

    def query(self, question: str) -> Answer:
        return self.rag_pipeline.answer(
            question=question,
            top_k=self.config.top_k,
        )

    def load_model(self) -> None:
        self.model.load()

    def unload_model(self) -> None:
        self.model.unload()
