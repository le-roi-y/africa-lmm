from __future__ import annotations

from typing import Any

from src.data.schemas import RetrievalResult


class ContextBuilder:
    """Construit un contexte RAG à partir des résultats textuels et multimodaux."""

    def build(
        self,
        results: list[RetrievalResult],
    ) -> str:
        if not results:
            return "No relevant context was found."

        contexts: list[str] = []

        for result in results:
            contexts.append(self._build_result_context(result))

        return "\n\n".join(contexts)

    def _build_result_context(
        self,
        result: RetrievalResult,
    ) -> str:
        chunk = result.chunk
        metadata = chunk.metadata

        lines = [
            (
                f"[Document: {chunk.document_id} | "
                f"Page: {chunk.page_number} | "
                f"Score: {result.score:.4f}]"
            ),
            "",
            chunk.text,
        ]

        images = metadata.get("images", [])
        if images:
            lines.extend(
                [
                    "",
                    "Images associated with this page:",
                ]
            )

            for image in images:
                lines.append(f"- {image}")

        tables = metadata.get("tables", [])
        if tables:
            lines.extend(
                [
                    "",
                    "Tables associated with this page:",
                ]
            )

            for table_index, table in enumerate(
                tables,
                start=1,
            ):
                lines.append(f"Table {table_index}:")
                lines.append(self._format_table(table))

        return "\n".join(lines)

    @staticmethod
    def _format_table(table: Any) -> str:
        if not isinstance(table, list):
            return str(table)

        rows: list[str] = []

        for row in table:
            if isinstance(row, (list, tuple)):
                rows.append(" | ".join(str(cell) for cell in row))
            else:
                rows.append(str(row))

        return "\n".join(rows)
