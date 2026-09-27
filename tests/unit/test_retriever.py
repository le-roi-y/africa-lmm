from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np

from src.data.schemas import DocumentChunk
from src.rag.retriever import InMemoryRetriever


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        page_number=1,
    )


def test_retriever_initializes_with_empty_index() -> None:
    with patch("src.rag.retriever.SentenceTransformer") as sentence_transformer:
        retriever = InMemoryRetriever("test-model")

    sentence_transformer.assert_called_once_with("test-model")
    assert retriever.chunks == []
    assert retriever.embeddings is None


def test_index_empty_chunks_does_nothing() -> None:
    with patch("src.rag.retriever.SentenceTransformer"):
        retriever = InMemoryRetriever("test-model")

    retriever.index([])

    assert retriever.chunks == []
    assert retriever.embeddings is None


def test_index_stores_chunks_and_embeddings() -> None:
    encoder = Mock()
    encoder.encode.return_value = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    with patch(
        "src.rag.retriever.SentenceTransformer",
        return_value=encoder,
    ):
        retriever = InMemoryRetriever("test-model")

    chunks = [
        make_chunk("chunk-1", "agriculture"),
        make_chunk("chunk-2", "climate"),
    ]

    retriever.index(chunks)

    assert retriever.chunks == chunks
    assert retriever.embeddings is not None
    np.testing.assert_array_equal(
        retriever.embeddings,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    encoder.encode.assert_called_once_with(
        ["agriculture", "climate"],
        normalize_embeddings=True,
    )


def test_index_appends_to_existing_embeddings() -> None:
    encoder = Mock()
    encoder.encode.side_effect = [
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
    ]

    with patch(
        "src.rag.retriever.SentenceTransformer",
        return_value=encoder,
    ):
        retriever = InMemoryRetriever("test-model")

    first_chunk = make_chunk("chunk-1", "agriculture")
    second_chunk = make_chunk("chunk-2", "climate")

    retriever.index([first_chunk])
    retriever.index([second_chunk])

    assert retriever.chunks == [first_chunk, second_chunk]
    assert retriever.embeddings is not None

    np.testing.assert_array_equal(
        retriever.embeddings,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    assert encoder.encode.call_count == 2


def test_retrieve_returns_empty_for_empty_index() -> None:
    with patch("src.rag.retriever.SentenceTransformer"):
        retriever = InMemoryRetriever("test-model")

    assert retriever.retrieve("agriculture") == []


def test_retrieve_ranks_results_by_similarity() -> None:
    encoder = Mock()

    encoder.encode.side_effect = [
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ]
        ),
        np.array([[1.0, 0.0]]),
    ]

    with patch(
        "src.rag.retriever.SentenceTransformer",
        return_value=encoder,
    ):
        retriever = InMemoryRetriever("test-model")

    chunks = [
        make_chunk("chunk-1", "agriculture"),
        make_chunk("chunk-2", "climate"),
        make_chunk("chunk-3", "agriculture and climate"),
    ]

    retriever.index(chunks)

    results = retriever.retrieve("agriculture", top_k=2)

    assert len(results) == 2

    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].rank == 1
    assert results[0].score == 1.0

    assert results[1].chunk.chunk_id == "chunk-3"
    assert results[1].rank == 2

    encoder.encode.assert_any_call(
        ["agriculture"],
        normalize_embeddings=True,
    )


def test_retrieve_respects_top_k() -> None:
    encoder = Mock()

    encoder.encode.side_effect = [
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ]
        ),
        np.array([[1.0, 0.0]]),
    ]

    with patch(
        "src.rag.retriever.SentenceTransformer",
        return_value=encoder,
    ):
        retriever = InMemoryRetriever("test-model")

    retriever.index(
        [
            make_chunk("chunk-1", "one"),
            make_chunk("chunk-2", "two"),
            make_chunk("chunk-3", "three"),
        ]
    )

    results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].rank == 1
