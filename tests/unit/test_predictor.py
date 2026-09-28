from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.data.schemas import Answer
from src.inference.predictor import Predictor


def test_predict_delegates_to_pipeline() -> None:
    pipeline = Mock()
    expected_answer = Answer(
        question="Quelle est la capitale du Cameroun ?",
        answer="Yaoundé",
    )
    pipeline.answer.return_value = expected_answer

    predictor = Predictor(pipeline)

    result = predictor.predict(
        question="Quelle est la capitale du Cameroun ?",
        top_k=7,
    )

    assert result is expected_answer

    pipeline.answer.assert_called_once_with(
        question="Quelle est la capitale du Cameroun ?",
        top_k=7,
        document_ids=None,
        conversation_context=None,
    )


def test_predict_uses_default_top_k() -> None:
    pipeline = Mock()
    expected_answer = Answer(
        question="Quelle est la capitale du Cameroun ?",
        answer="Yaoundé",
    )
    pipeline.answer.return_value = expected_answer

    predictor = Predictor(pipeline)

    result = predictor.predict("Quelle est la capitale du Cameroun ?")

    assert result is expected_answer

    pipeline.answer.assert_called_once_with(
        question="Quelle est la capitale du Cameroun ?",
        top_k=5,
        document_ids=None,
        conversation_context=None,
    )


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\t",
        "\n",
        "   \t\n   ",
    ],
)
def test_predict_rejects_empty_question(question: str) -> None:
    pipeline = Mock()
    predictor = Predictor(pipeline)

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        predictor.predict(question)

    pipeline.answer.assert_not_called()
