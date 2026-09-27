from __future__ import annotations

from pydantic import BaseModel

from src.data.schemas import Answer

from .metrics import exact_match, token_f1


class EvaluationReport(BaseModel):
    exact_match: float
    f1: float
    number_of_examples: int


class Evaluator:
    """Evaluate generated answers against reference answers."""

    def evaluate(
        self,
        predictions: list[Answer],
        references: list[str],
    ) -> EvaluationReport:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        if not predictions:
            return EvaluationReport(
                exact_match=0.0,
                f1=0.0,
                number_of_examples=0,
            )

        em_scores = [
            exact_match(prediction.answer, reference)
            for prediction, reference in zip(predictions, references, strict=True)
        ]

        f1_scores = [
            token_f1(prediction.answer, reference)
            for prediction, reference in zip(predictions, references, strict=True)
        ]

        return EvaluationReport(
            exact_match=sum(em_scores) / len(em_scores),
            f1=sum(f1_scores) / len(f1_scores),
            number_of_examples=len(predictions),
        )
