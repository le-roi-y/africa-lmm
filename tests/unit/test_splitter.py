import pytest

from src.data.schemas import Document, DocumentMetadata, DocumentPage
from src.data.splitter import TextSplitter


def make_document(text: str = "") -> Document:
    return Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="document.txt",
        ),
        text=text,
    )


def test_splitter_creates_chunks():
    document = make_document("A" * 250)

    splitter = TextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = splitter.split(document)

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_splitter_preserves_page_number():
    document = Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="document.pdf",
        ),
        pages=[
            DocumentPage(
                page_number=3,
                text="Contenu de la page trois.",
            )
        ],
    )

    chunks = TextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    ).split(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3


def test_empty_document_returns_no_chunks():
    document = make_document("")

    chunks = TextSplitter().split(document)

    assert chunks == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        TextSplitter(
            chunk_size=100,
            chunk_overlap=100,
        )
