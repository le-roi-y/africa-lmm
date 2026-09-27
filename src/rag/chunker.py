from __future__ import annotations

from src.data.schemas import Document, DocumentChunk
from src.data.splitter import TextSplitter


class Chunker:

    def __init__(
        self,
        splitter: TextSplitter | None = None,
    ) -> None:

        self.splitter = splitter or TextSplitter()

    def chunk(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        return self.splitter.split(document)
