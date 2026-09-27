from __future__ import annotations

from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from src.data.schemas import RetrievalResult


class Reranker(ABC):
    """Interface abstraite pour le reranking des résultats RAG."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Reranker multilingue basé sur un Cross-Encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")

        self.model_name = model_name
        self.max_length = max_length
        self.model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        """Load the Cross-Encoder only when reranking is required."""
        if self.model is None:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )

        return self.model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not results:
            return []

        pairs = [
            (query, result.chunk.text)
            for result in results
        ]

        model = self._get_model()

        scores = model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        reranked = [
            result.model_copy(
                update={"score": float(score)}
            )
            for result, score in zip(
                results,
                scores,
                strict=True,
            )
        ]

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return [
            result.model_copy(
                update={"rank": rank}
            )
            for rank, result in enumerate(
                reranked,
                start=1,
            )
        ]


class ScoreReranker(Reranker):
    """Reranker simple basé sur le score de retrieval."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        del query

        return [
            result.model_copy(
                update={"rank": rank}
            )
            for rank, result in enumerate(
                sorted(
                    results,
                    key=lambda result: result.score,
                    reverse=True,
                ),
                start=1,
            )
        ]
