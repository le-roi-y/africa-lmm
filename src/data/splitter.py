from __future__ import annotations

from typing import Any

from .schemas import Document, DocumentChunk


class TextSplitter:
    """Découpe un document en chunks avec contexte multimodal."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        document: Document,
    ) -> list[DocumentChunk]:
        if document.pages:
            chunks: list[DocumentChunk] = []

            for page in document.pages:
                chunks.extend(
                    self._split_text(
                        document=document,
                        text=page.text,
                        page_number=page.page_number,
                        images=page.images,
                        tables=page.tables,
                    )
                )

            return chunks

        return self._split_text(
            document=document,
            text=document.text,
            page_number=None,
            images=[],
            tables=[],
        )

    def _split_text(
        self,
        document: Document,
        text: str,
        page_number: int | None,
        images: list[str],
        tables: list[Any],
    ) -> list[DocumentChunk]:
        text = text.strip()

        if not text:
            return []

        chunks: list[DocumentChunk] = []

        start = 0
        index = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{document.metadata.document_id}" f"_{page_number or 0}" f"_{index}"
                        ),
                        document_id=document.metadata.document_id,
                        text=chunk_text,
                        page_number=page_number,
                        metadata={
                            "language": document.metadata.language,
                            "country": document.metadata.country,
                            "domain": document.metadata.domain,
                            "images": images,
                            "tables": tables,
                        },
                    )
                )

                index += 1

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        return chunks
