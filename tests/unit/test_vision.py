from pathlib import Path

from PIL import Image

from src.data.schemas import Document, DocumentMetadata, DocumentPage
from src.vision.document_vision import DocumentVision
from src.vision.image_processor import ImageProcessor


def test_image_processor_load(tmp_path: Path):
    image_path = tmp_path / "test.png"

    image = Image.new("RGB", (100, 50))
    image.save(image_path)

    processor = ImageProcessor()
    loaded = processor.load(image_path)

    assert loaded.size == (100, 50)


def test_image_processor_resize():
    image = Image.new("RGB", (4000, 2000))

    resized = ImageProcessor().resize(image, max_size=1000)

    assert max(resized.size) <= 1000


def test_document_vision():
    document = Document(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="report.pdf",
        ),
        pages=[
            DocumentPage(
                page_number=1,
                images=["figure.png"],
            ),
            DocumentPage(
                page_number=2,
                tables=[{"rows": 2}],
            ),
        ],
    )

    elements = DocumentVision().analyze_document(document)

    assert len(elements) == 2
    assert elements[0].element_type == "image"
    assert elements[0].page_number == 1
    assert elements[1].element_type == "table"
    assert elements[1].page_number == 2
