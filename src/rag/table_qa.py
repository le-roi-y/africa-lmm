from __future__ import annotations

import re
from typing import Any


class TableQA:
    """Répond aux questions simples portant sur des tableaux structurés."""

    def answer(
        self,
        question: str,
        tables: list[Any],
        column_question: str | None = None,
    ) -> str | None:
        """
        Answer a table question.

        The current question identifies the requested row/entity.
        An optional contextual question can provide missing information
        such as a year or column.
        """
        for table in tables:
            result = self._answer_from_table(
                question=question,
                column_question=column_question or question,
                table=table,
            )

            if result is not None:
                return result

        return None

    def _answer_from_table(
        self,
        question: str,
        column_question: str,
        table: Any,
    ) -> str | None:
        if not isinstance(table, list) or len(table) < 2:
            return None

        rows = [row for row in table if isinstance(row, (list, tuple))]

        if len(rows) < 2:
            return None

        headers = [str(value).strip() for value in rows[0]]

        column_index = self._find_column(
            column_question.lower(),
            headers,
        )

        if column_index is None:
            return None

        resolved_question = column_question.lower()

        for row in rows[1:]:
            if not row:
                continue

            label = str(row[0]).strip()

            if label.lower() in resolved_question:
                if column_index >= len(row):
                    return None

                return str(row[column_index]).strip()

        return None

    @staticmethod
    def _find_column(
        question: str,
        headers: list[str],
    ) -> int | None:
        for index, header in enumerate(headers):
            if header.lower() in question:
                return index

        years = re.findall(r"\b(?:19|20)\d{2}\b", question)

        if years:
            year = years[0]

            for index, header in enumerate(headers):
                if header == year:
                    return index

        return None
