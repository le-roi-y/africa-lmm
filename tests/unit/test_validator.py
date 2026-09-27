import pytest

from src.data.schemas import Document, DocumentMetadata
from src.data.validator import DocumentValidationError, DocumentValidator


def test_valid_document():
    document = Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="document.txt",
        ),
        text="Contenu valide.",
    )

    DocumentValidator().validate(document)


def test_document_without_id():
    document = Document(
        metadata=DocumentMetadata(
            document_id="",
            filename="document.txt",
        ),
        text="Contenu.",
    )

    with pytest.raises(DocumentValidationError):
        DocumentValidator().validate(document)


def test_document_without_filename():
    document = Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="",
        ),
        text="Contenu.",
    )

    with pytest.raises(DocumentValidationError):
        DocumentValidator().validate(document)


def test_empty_document():
    document = Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="document.txt",
        )
    )

    with pytest.raises(DocumentValidationError):
        DocumentValidator().validate(document)
