from pathlib import Path

import fitz
import pytest

from src.data.loader import PDFDocumentLoader, TextDocumentLoader


def test_text_document_loader(tmp_path: Path):
    file_path = tmp_path / "document.txt"
    file_path.write_text(
        "Bonjour AFRICA-LMM.\nVoici un document de test.",
        encoding="utf-8",
    )

    loader = TextDocumentLoader()
    document = loader.load(file_path)

    assert document.metadata.document_id == "document"
    assert document.metadata.filename == "document.txt"
    assert "AFRICA-LMM" in document.text


def test_text_document_loader_missing_file(tmp_path: Path):
    loader = TextDocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "missing.txt")


def test_text_document_loader_rejects_directory(tmp_path: Path):
    loader = TextDocumentLoader()

    with pytest.raises(ValueError):
        loader.load(tmp_path)


def test_pdf_document_loader(tmp_path: Path):
    pdf_path = tmp_path / "report.pdf"

    pdf = fitz.open()

    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "AFRICA-LMM PDF test.\nAgriculture au Cameroun.",
    )

    pdf.save(pdf_path)
    pdf.close()

    loader = PDFDocumentLoader()
    document = loader.load(pdf_path)

    assert document.metadata.document_id == "report"
    assert document.metadata.filename == "report.pdf"
    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert "AFRICA-LMM" in document.pages[0].text


def test_pdf_document_loader_rejects_non_pdf(tmp_path: Path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("test", encoding="utf-8")

    loader = PDFDocumentLoader()

    with pytest.raises(ValueError):
        loader.load(file_path)
