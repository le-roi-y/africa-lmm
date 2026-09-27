from src.data.schemas import DocumentPage
from src.ocr.engine import OCREngine, OCRResult
from src.ocr.processor import OCRProcessor


class FakeOCREngine(OCREngine):
    def recognize(
        self,
        image_path: str,
        language: str | None = None,
    ) -> OCRResult:
        return OCRResult(
            text=f"OCR result for {image_path}",
            confidence=0.99,
            language=language or "eng",
        )


def test_ocr_processor_image():
    processor = OCRProcessor(FakeOCREngine())

    result = processor.process_image(
        "document.png",
        language="fra",
    )

    assert result.text == "OCR result for document.png"
    assert result.confidence == 0.99
    assert result.language == "fra"


def test_ocr_processor_page():
    processor = OCRProcessor(FakeOCREngine())

    page = DocumentPage(
        page_number=1,
        text="",
        images=["image1.png", "image2.png"],
    )

    results = processor.process_page(
        page,
        language="fra",
    )

    assert len(results) == 2
    assert results[0].language == "fra"
    assert results[1].language == "fra"
