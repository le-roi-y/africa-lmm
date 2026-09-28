from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pymupdf as fitz

from src.data.schemas import Document, DocumentMetadata, DocumentPage
from src.ocr.engine import TesseractEngine


class DocumentEngine:
    """
    Moteur intelligent de traitement des documents PDF.

    Pipeline :

        PDF
         ├── Texte natif
         ├── OCR si nécessaire
         ├── Extraction des images
         └── Extraction des tableaux
    """

    def __init__(
        self,
        min_text_chars: int = 20,
        ocr_language: str = "fra+eng",
        dpi: int = 200,
        output_dir: str | Path = "data/processed",
    ) -> None:
        if min_text_chars < 0:
            raise ValueError("min_text_chars must be >= 0.")

        if dpi <= 0:
            raise ValueError("dpi must be greater than 0.")

        self.min_text_chars = min_text_chars
        self.ocr_language = ocr_language
        self.dpi = dpi
        self.output_dir = Path(output_dir)
        self.ocr_engine = TesseractEngine()

    def load_pdf(self, path: str | Path) -> Document:
        """Load a PDF with text extraction, OCR, images and tables."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        document_output_dir = self.output_dir / path.stem
        image_output_dir = document_output_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        pdf = fitz.open(str(path))

        try:
            pages: list[DocumentPage] = []
            full_text: list[str] = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if len(text) < self.min_text_chars:
                    text = self._ocr_page(
                        page=page,
                        page_number=page_number,
                    )

                images = self._extract_images(
                    page=page,
                    page_number=page_number,
                    output_dir=image_output_dir,
                )

                tables = self._extract_tables(page)

                document_page = DocumentPage(
                    page_number=page_number,
                    text=text,
                    images=images,
                    tables=tables,
                )

                pages.append(document_page)
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

    def _ocr_page(
        self,
        page: fitz.Page,
        page_number: int,
    ) -> str:
        """Render a PDF page and run OCR on it."""
        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / f"page_{page_number}.png"

            pixmap.save(str(image_path))

            result = self.ocr_engine.recognize(
                image_path=str(image_path),
                language=self.ocr_language,
            )

            return result.text.strip()

    @staticmethod
    def _extract_images(
        page: fitz.Page,
        page_number: int,
        output_dir: Path,
    ) -> list[str]:
        """Extract embedded images from a PDF page."""
        image_paths: list[str] = []

        for image_index, image_info in enumerate(
            page.get_images(full=True),
            start=1,
        ):
            xref = image_info[0]

            try:
                image = page.parent.extract_image(xref)
            except Exception:
                continue

            image_bytes = image.get("image")
            image_ext = image.get("ext", "png")

            if not image_bytes:
                continue

            image_path = output_dir / f"page_{page_number}_image_{image_index}.{image_ext}"

            image_path.write_bytes(image_bytes)
            image_paths.append(str(image_path))

        return image_paths

    @staticmethod
    def _extract_tables(
        page: fitz.Page,
    ) -> list[Any]:
        """Extract tables detected by PyMuPDF."""
        try:
            table_finder = page.find_tables()
        except (AttributeError, RuntimeError):
            return []

        tables: list[Any] = []

        for table in table_finder.tables:
            try:
                extracted = table.extract()
            except Exception:
                continue

            if extracted:
                tables.append(extracted)

        return tables
