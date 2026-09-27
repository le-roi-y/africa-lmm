from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Result returned by an OCR engine."""

    text: str
    confidence: float | None = None
    language: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class OCREngine(ABC):
    """Abstract interface for OCR engines."""

    @abstractmethod
    def recognize(
        self,
        image_path: str,
        language: str | None = None,
    ) -> OCRResult:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    """OCR implementation using Tesseract."""

    def recognize(
        self,
        image_path: str,
        language: str | None = None,
    ) -> OCRResult:
        import pytesseract
        from PIL import Image

        lang = language or "eng"

        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(
                image,
                lang=lang,
            )

        return OCRResult(
            text=text,
            language=lang,
        )
