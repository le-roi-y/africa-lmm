from __future__ import annotations

from src.data.schemas import Answer
from src.rag.pipeline import RAGPipeline


class Predictor:

    def __init__(
        self,
        pipeline: RAGPipeline,
    ) -> None:

        self.pipeline = pipeline

    def predict(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        conversation_context: str | None = None,
    ) -> Answer:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        return self.pipeline.answer(
            question=question,
            top_k=top_k,
            document_ids=document_ids,
            conversation_context=conversation_context,
        )
