from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import Document, DocumentMetadata, DocumentPage


class DocumentLoader(ABC):
    """Interface de base pour les chargeurs de documents."""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        raise NotImplementedError


class TextDocumentLoader(DocumentLoader):
    """Charge un fichier texte."""

    def load(self, path: str | Path) -> Document:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        text = path.read_text(encoding="utf-8")

        metadata = DocumentMetadata(
            document_id=path.stem,
            filename=path.name,
        )

        return Document(
            metadata=metadata,
            text=text,
        )


class PDFDocumentLoader(DocumentLoader):
    """Charge un document PDF avec PyMuPDF."""

    def load(self, path: str | Path) -> Document:
        import fitz

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        pdf = fitz.open(str(path))

        try:
            pages: list[DocumentPage] = []
            full_text: list[str] = []

            for index, page in enumerate(pdf):
                text = page.get_text("text")

                pages.append(
                    DocumentPage(
                        page_number=index + 1,
                        text=text,
                    )
                )

                full_text.append(text)

            metadata = DocumentMetadata(
                document_id=path.stem,
                filename=path.name,
            )

            return Document(
                metadata=metadata,
                text="\n".join(full_text),
                pages=pages,
            )

        finally:
            pdf.close()
