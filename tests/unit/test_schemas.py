from src.data.schemas import (
    Answer,
    Citation,
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentPage,
    RetrievalResult,
)


def test_document_metadata():
    metadata = DocumentMetadata(
        document_id="doc-001",
        filename="report.pdf",
        language="fr",
        country="CM",
        domain="agriculture",
        year=2026,
    )

    assert metadata.document_id == "doc-001"
    assert metadata.filename == "report.pdf"
    assert metadata.language == "fr"
    assert metadata.country == "CM"
    assert metadata.domain == "agriculture"
    assert metadata.year == 2026


def test_document_page():
    page = DocumentPage(
        page_number=1,
        text="Contenu de la page.",
    )

    assert page.page_number == 1
    assert page.text == "Contenu de la page."
    assert page.images == []
    assert page.tables == []


def test_document():
    metadata = DocumentMetadata(
        document_id="doc-001",
        filename="report.pdf",
    )

    document = Document(
        metadata=metadata,
        text="Contenu du document.",
    )

    assert document.metadata.document_id == "doc-001"
    assert document.text == "Contenu du document."


def test_document_chunk():
    chunk = DocumentChunk(
        chunk_id="doc-001_1_0",
        document_id="doc-001",
        text="Un morceau du document.",
        page_number=1,
    )

    assert chunk.chunk_id == "doc-001_1_0"
    assert chunk.document_id == "doc-001"
    assert chunk.page_number == 1


def test_retrieval_result():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Information importante.",
    )

    result = RetrievalResult(
        chunk=chunk,
        score=0.95,
        rank=1,
    )

    assert result.score == 0.95
    assert result.rank == 1
    assert result.chunk.chunk_id == "chunk-1"


def test_answer_and_citation():
    citation = Citation(
        document_id="doc-1",
        page_number=4,
        chunk_id="chunk-1",
    )

    answer = Answer(
        question="Quelle est la production agricole ?",
        answer="La production est de 100 tonnes.",
        citations=[citation],
    )

    assert answer.question == "Quelle est la production agricole ?"
    assert answer.answer == "La production est de 100 tonnes."
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 4
