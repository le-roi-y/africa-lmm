from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def exact_match(
    prediction: str,
    reference: str,
) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(
    prediction: str,
    reference: str,
) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts = {token: prediction_tokens.count(token) for token in set(prediction_tokens)}
    reference_counts = {token: reference_tokens.count(token) for token in set(reference_tokens)}

    common = sum(
        min(
            prediction_counts.get(token, 0),
            reference_counts.get(token, 0),
        )
        for token in set(prediction_counts)
    )

    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def recall_at_k(
    rank: int | None,
    k: int,
) -> float:
    if rank is None:
        return 0.0

    return float(rank <= k)


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0

    return 1.0 / rank


def citation_accuracy(
    correct: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    return correct / total
