import pytest

from src.data.schemas import Answer
from src.evaluation.evaluator import Evaluator


def make_answer(text: str) -> Answer:
    return Answer(
        question="Question",
        answer=text,
    )


def test_evaluator():
    predictions = [
        make_answer("Paris"),
        make_answer("Londres"),
    ]

    references = [
        "Paris",
        "Lyon",
    ]

    report = Evaluator().evaluate(predictions, references)

    assert report.number_of_examples == 2
    assert report.exact_match == 0.5
    assert 0.0 <= report.f1 <= 1.0


def test_evaluator_empty():
    report = Evaluator().evaluate([], [])

    assert report.number_of_examples == 0
    assert report.exact_match == 0.0
    assert report.f1 == 0.0


def test_evaluator_length_mismatch():
    predictions = [make_answer("Paris")]

    with pytest.raises(ValueError):
        Evaluator().evaluate(predictions, [])
