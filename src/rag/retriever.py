from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.data.schemas import (
    DocumentChunk,
    RetrievalResult,
)


class Retriever(ABC):

    @abstractmethod
    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class InMemoryRetriever(Retriever):

    def __init__(
        self,
        embedding_model: str,
    ) -> None:

        self.encoder = SentenceTransformer(embedding_model)

        self.chunks: list[DocumentChunk] = []
        self.embeddings: NDArray[Any] | None = None

    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:

        if not chunks:
            return

        self.chunks.extend(chunks)

        texts = [chunk.text for chunk in chunks]

        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
        )

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not self.chunks or self.embeddings is None:
            return []

        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        ]
