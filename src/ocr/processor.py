from __future__ import annotations

from src.data.schemas import DocumentPage

from .engine import OCREngine, OCRResult


class OCRProcessor:
    """Coordinate OCR operations for images and document pages."""

    def __init__(self, engine: OCREngine) -> None:
        self.engine = engine

    def process_image(
        self,
        image_path: str,
        language: str | None = None,
    ) -> OCRResult:
        return self.engine.recognize(
            image_path=image_path,
            language=language,
        )

    def process_page(
        self,
        page: DocumentPage,
        language: str | None = None,
    ) -> list[OCRResult]:
        return [
            self.process_image(
                image_path=image_path,
                language=language,
            )
            for image_path in page.images
        ]
