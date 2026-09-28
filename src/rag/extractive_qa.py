from __future__ import annotations

import re

from sentence_transformers import CrossEncoder

from src.data.schemas import RetrievalResult


class ExtractiveQA:
    """Extract answers explicitly stated in retrieved document chunks."""

    def __init__(
        self,
        model: CrossEncoder,
        score_threshold: float = 0.35,
        min_sentence_length: int = 20,
        max_sentence_length: int = 500,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1.")

        if min_sentence_length <= 0:
            raise ValueError("min_sentence_length must be greater than 0.")

        if max_sentence_length < min_sentence_length:
            raise ValueError(
                "max_sentence_length must be greater than or equal " "to min_sentence_length."
            )

        self.model = model
        self.score_threshold = score_threshold
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length

    def answer(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str | None:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        candidates = self._build_candidates(results)

        if not candidates:
            return None

        pairs = [(question, candidate) for candidate in candidates]

        scores = self.model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        best_candidate, best_score = ranked[0]

        if float(best_score) < self.score_threshold:
            return None

        return best_candidate

    def _build_candidates(
        self,
        results: list[RetrievalResult],
    ) -> list[str]:
        candidates: list[str] = []

        for result in results:
            text = result.chunk.text.strip()

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            for sentence in sentences:
                sentence = " ".join(sentence.split())

                if not (self.min_sentence_length <= len(sentence) <= self.max_sentence_length):
                    continue

                candidates.append(sentence)

        return list(dict.fromkeys(candidates))
