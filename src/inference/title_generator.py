from __future__ import annotations

import re


QUESTION_STARTERS = {
    "quelle est",
    "quels sont",
    "quelles sont",
    "quel est",
    "pourquoi",
    "comment",
    "où",
    "quand",
    "qui",
    "peux-tu",
    "peut-tu",
    "peux tu",
    "peut tu",
    "donne-moi",
    "donne moi",
    "explique-moi",
    "explique moi",
    "résume",
    "resume",
}


def generate_conversation_title(question: str, max_length: int = 60) -> str:
    """Génère un titre court et lisible à partir de la première question."""

    text = re.sub(r"\s+", " ", question.strip())

    if not text:
        return "Nouvelle conversation"

    # Supprime la ponctuation finale.
    text = text.rstrip(" ?!.,;:")

    # Retire les formulations interrogatives courantes.
    lowered = text.lower()

    for starter in sorted(QUESTION_STARTERS, key=len, reverse=True):
        if lowered.startswith(starter + " "):
            text = text[len(starter) :].strip()
            break

    # Retire quelques formulations trop longues.
    prefixes = (
        "est-ce que ",
        "dis-moi ",
        "dites-moi ",
        "je voudrais savoir ",
        "j'aimerais savoir ",
        "j'aimerais connaître ",
    )

    lowered = text.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    if not text:
        text = question.strip().rstrip(" ?!.,;:")

    # Première lettre en majuscule.
    text = text[0].upper() + text[1:] if text else "Nouvelle conversation"

    # Limite proprement la longueur.
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0].rstrip(" ,;:-")

    return text
