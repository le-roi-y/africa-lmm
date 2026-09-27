from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from sentence_transformers import SentenceTransformer

from src.data.schemas import DocumentChunk, RetrievalResult
from src.rag.retriever import Retriever


class QdrantRetriever(Retriever):
    """Persistent vector retriever backed by Qdrant."""

    def __init__(
        self,
        embedding_model: str,
        collection_name: str = "africa_lmm_documents",
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.client = QdrantClient(
            host=host,
            port=port,
        )

        self.encoder: SentenceTransformer | None = None
        self.vector_size: int | None = None

    def _load_encoder(self) -> SentenceTransformer:
        """Load the embedding model only when it is actually needed."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(
                self.embedding_model_name,
            )

            vector_size = self.encoder.get_embedding_dimension()

            if vector_size is None:
                raise RuntimeError(
                    "Unable to determine embedding dimension."
                )

            self.vector_size = vector_size
            self._ensure_collection()

        return self.encoder

    def get_encoder(self) -> SentenceTransformer:
        """Return the embedding encoder, loading it lazily if needed."""
        return self._load_encoder()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if self.vector_size is None:
            raise RuntimeError(
                "Embedding dimension must be known before creating "
                "the collection."
            )

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Encode and persist document chunks in Qdrant."""
        if not chunks:
            return

        encoder = self._load_encoder()

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = encoder.encode(
            texts,
            normalize_embeddings=True,
        )

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            points.append(
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "metadata": chunk.metadata,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def retrieve_document(
        self,
        document_id: str,
        limit: int = 100,
    ) -> list[RetrievalResult]:
        """Retrieve indexed chunks belonging to one document."""
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                ],
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for rank, record in enumerate(records, start=1):
            payload: dict[str, Any] = record.payload or {}

            chunk = DocumentChunk(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                text=str(payload["text"]),
                page_number=payload.get("page_number"),
                metadata=payload.get("metadata", {}),
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=1.0,
                    rank=rank,
                )
            )

        results.sort(
            key=lambda result: (
                result.chunk.page_number or 0,
                result.chunk.chunk_id,
            )
        )

        for rank, result in enumerate(results, start=1):
            result.rank = rank

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the top semantic candidates from Qdrant."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        encoder = self._load_encoder()

        query_embedding = encoder.encode(
            query,
            normalize_embeddings=True,
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

        retrieval_results = []

        for rank, result in enumerate(results, start=1):
            payload: dict[str, Any] = result.payload or {}

            chunk = DocumentChunk(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                text=str(payload["text"]),
                page_number=payload.get("page_number"),
                metadata=payload.get("metadata", {}),
            )

            retrieval_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(result.score),
                    rank=rank,
                )
            )

        return retrieval_results
