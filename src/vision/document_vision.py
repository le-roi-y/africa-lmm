from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.data.schemas import Document


class VisualElement(BaseModel):
    """Element visuel extrait d'un document."""

    element_type: str
    page_number: int
    bounding_box: (
        tuple[
            float,
            float,
            float,
            float,
        ]
        | None
    ) = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentVision:
    """Analyse les éléments visuels d'un document."""

    def analyze_document(
        self,
        document: Document,
    ) -> list[VisualElement]:
        elements: list[VisualElement] = []

        for page in document.pages:
            self._add_images(
                document=document,
                page=page,
                elements=elements,
            )

            self._add_tables(
                document=document,
                page=page,
                elements=elements,
            )

        return elements

    @staticmethod
    def _add_images(
        document: Document,
        page: Any,
        elements: list[VisualElement],
    ) -> None:
        """Convert extracted images into visual elements."""
        for image_index, image_path in enumerate(
            page.images,
            start=1,
        ):
            elements.append(
                VisualElement(
                    element_type="image",
                    page_number=page.page_number,
                    content=image_path,
                    metadata={
                        "document_id": document.metadata.document_id,
                        "filename": document.metadata.filename,
                        "image_index": image_index,
                        "source": "pdf_extraction",
                    },
                )
            )

    @staticmethod
    def _add_tables(
        document: Document,
        page: Any,
        elements: list[VisualElement],
    ) -> None:
        """Convert extracted tables into visual elements."""
        for table_index, table in enumerate(
            page.tables,
            start=1,
        ):
            elements.append(
                VisualElement(
                    element_type="table",
                    page_number=page.page_number,
                    content=None,
                    metadata={
                        "document_id": document.metadata.document_id,
                        "filename": document.metadata.filename,
                        "table_index": table_index,
                        "rows": table,
                        "source": "pdf_table_extraction",
                    },
                )
            )
