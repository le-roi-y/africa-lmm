from __future__ import annotations

from unittest.mock import Mock

from src.data.schemas import Document, DocumentMetadata
from src.inference.pipeline import InferenceConfig, InferencePipeline


def make_document() -> Document:
    return Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="rapport.txt",
        ),
        text="L'agriculture représente un secteur important de l'économie africaine.",
    )


def test_default_config() -> None:
    config = InferenceConfig()

    assert config.top_k == 5
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200


def test_custom_config() -> None:
    config = InferenceConfig(
        top_k=10,
        chunk_size=500,
        chunk_overlap=100,
    )

    rag_pipeline = Mock()
    model = Mock()

    pipeline = InferencePipeline(
        rag_pipeline=rag_pipeline,
        model=model,
        config=config,
    )

    assert pipeline.config is config
    assert pipeline.splitter.chunk_size == 500
    assert pipeline.splitter.chunk_overlap == 100


def test_index_document_splits_and_indexes_document() -> None:
    rag_pipeline = Mock()
    model = Mock()

    pipeline = InferencePipeline(
        rag_pipeline=rag_pipeline,
        model=model,
    )

    document = make_document()

    pipeline.index_document(document)

    rag_pipeline.retriever.index.assert_called_once()

    indexed_chunks = rag_pipeline.retriever.index.call_args.args[0]

    assert len(indexed_chunks) == 1
    assert indexed_chunks[0].document_id == "doc-1"
    assert indexed_chunks[0].text == document.text


def test_query_uses_configured_top_k() -> None:
    rag_pipeline = Mock()
    model = Mock()

    expected_answer = Mock()
    rag_pipeline.answer.return_value = expected_answer

    pipeline = InferencePipeline(
        rag_pipeline=rag_pipeline,
        model=model,
        config=InferenceConfig(top_k=7),
    )

    result = pipeline.query("Quelle est l'importance de l'agriculture ?")

    assert result is expected_answer

    rag_pipeline.answer.assert_called_once_with(
        question="Quelle est l'importance de l'agriculture ?",
        top_k=7,
    )


def test_load_model_delegates_to_model() -> None:
    rag_pipeline = Mock()
    model = Mock()

    pipeline = InferencePipeline(
        rag_pipeline=rag_pipeline,
        model=model,
    )

    pipeline.load_model()

    model.load.assert_called_once_with()


def test_unload_model_delegates_to_model() -> None:
    rag_pipeline = Mock()
    model = Mock()

    pipeline = InferencePipeline(
        rag_pipeline=rag_pipeline,
        model=model,
    )

    pipeline.unload_model()

    model.unload.assert_called_once_with()
