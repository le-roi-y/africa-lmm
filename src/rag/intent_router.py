from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class Intent:
    DOCUMENT_QA = "document_qa"
    DOCUMENT_SUMMARY = "document_summary"
    DOCUMENT_COMPARISON = "document_comparison"
    DOCUMENT_OVERVIEW = "document_overview"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RouterContext:
    selected_document_count: int = 0
    selected_document_ids: tuple[str, ...] = ()

    @property
    def has_selected_documents(self) -> bool:
        return self.selected_document_count > 0


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    reason: str
    ambiguous: bool = False


class IntentRouter:
    """
    Semantic intent router.

    Uses the same multilingual embedding space as the RAG retriever.
    The LLM is intentionally not responsible for producing structured JSON.
    """

    MIN_CONFIDENCE = 0.55
    MIN_MARGIN = 0.08

    _EXAMPLES: dict[str, tuple[str, ...]] = {
        Intent.DOCUMENT_QA: (
            "Quelle est la production de cacao en 2024 ?",
            "Combien de tonnes de maïs ont été produites ?",
            "Quand cette politique a-t-elle été mise en œuvre ?",
            "Où se situe cette région ?",
            "Pourquoi cette mesure a-t-elle été adoptée ?",
            "Quel est le montant indiqué dans le rapport ?",
            "Qui est mentionné dans ce document ?",
            "What is the production reported in the document?",
            "When was this policy implemented?",
            "How much was produced?",
        ),
        Intent.DOCUMENT_SUMMARY: (
            "Résume ce document.",
            "Donne-moi un résumé de ce rapport.",
            "Quelles sont les idées essentielles de ce document ?",
            "Donne-moi les points clés du rapport.",
            "Synthétise les informations importantes.",
            "Présente les principales conclusions du document.",
            "Donne-moi l'essentiel de ce rapport.",
            "What are the main points of this document?",
            "Summarize the report.",
            "Give me the key findings.",
        ),
        Intent.DOCUMENT_COMPARISON: (
            "Compare ces deux documents.",
            "Quelles sont les différences entre ces deux rapports ?",
            "En quoi ces documents sont-ils différents ?",
            "Quels changements observes-tu entre les deux périodes ?",
            "Compare les résultats présentés dans les rapports.",
            "Quelles sont les similitudes et les différences ?",
            "Comment les deux rapports se comparent-ils ?",
            "What are the differences between these two reports?",
            "Compare the two documents.",
            "What changed between the two periods?",
        ),
        Intent.DOCUMENT_OVERVIEW: (
            "Que contient globalement ce document ?",
            "Donne-moi une vue d'ensemble de ce rapport.",
            "De quoi parle ce document ?",
            "Présente-moi ce document dans son ensemble.",
            "Comment ce rapport est-il organisé ?",
            "Quels sont les grands thèmes abordés dans ce document ?",
            "Aide-moi à comprendre la structure de ce rapport.",
            "What is this document about?",
            "Give me an overview of this report.",
            "What are the main topics covered?",
        ),
    }

    def __init__(
        self,
        model: BaseModel | None = None,
        encoder: SentenceTransformer | None = None,
        encoder_provider=None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.encoder_provider = encoder_provider
        self._prototype_embeddings = None

        if self.encoder is None and self.encoder_provider is None:
            raise ValueError(
                "IntentRouter requires an embedding encoder or provider."
            )

    def _get_encoder(self) -> SentenceTransformer:
        """Return the embedding encoder, loading it lazily if necessary."""
        if self.encoder is None:
            if self.encoder_provider is None:
                raise RuntimeError(
                    "IntentRouter embedding encoder is unavailable."
                )

            self.encoder = self.encoder_provider()

        return self.encoder

    def _get_prototype_embeddings(self) -> dict[str, np.ndarray]:
        """Build intent prototypes lazily on first use."""
        if self._prototype_embeddings is None:
            self._prototype_embeddings = self._build_prototypes()

        return self._prototype_embeddings

    def route(
        self,
        question: str,
        context: RouterContext | None = None,
    ) -> IntentResult:
        question = question.strip()
        context = context or RouterContext()

        if not question:
            return IntentResult(
                intent=Intent.DOCUMENT_QA,
                confidence=1.0,
                reason="empty_question",
            )

        try:
            encoder = self._get_encoder()

            query_embedding = encoder.encode(
                question,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.warning("Intent embedding failure: %s", exc)
            return self._fallback("embedding_failure")

        prototype_embeddings = self._get_prototype_embeddings()

        scores = {
            intent: float(np.dot(query_embedding, embedding))
            for intent, embedding in prototype_embeddings.items()
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_intent, best_score = ranked[0]
        second_score = ranked[1][1]
        margin = best_score - second_score

        confidence = self._confidence(
            best_score=best_score,
            margin=margin,
        )

        if (
            best_intent == Intent.DOCUMENT_COMPARISON
            and context.selected_document_count == 1
        ):
            return IntentResult(
                intent=Intent.AMBIGUOUS,
                confidence=confidence,
                reason="comparison_requested_but_only_one_document_selected",
                ambiguous=True,
            )

        if (
            best_score >= self.MIN_CONFIDENCE
            and margin >= self.MIN_MARGIN
        ):
            return IntentResult(
                intent=best_intent,
                confidence=confidence,
                reason="semantic_match",
            )

        return IntentResult(
            intent=Intent.AMBIGUOUS,
            confidence=confidence,
            reason="low_confidence_or_margin",
            ambiguous=True,
        )

    def _build_prototypes(self) -> dict[str, np.ndarray]:
        encoder = self._get_encoder()
        prototypes: dict[str, np.ndarray] = {}

        for intent, examples in self._EXAMPLES.items():
            embeddings = encoder.encode(
                list(examples),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)

            if norm == 0:
                raise ValueError(
                    f"Invalid prototype embedding for intent: {intent}"
                )

            prototypes[intent] = centroid / norm

        return prototypes

    @staticmethod
    def _confidence(
        best_score: float,
        margin: float,
    ) -> float:
        score_component = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        margin_component = max(0.0, min(1.0, margin / 0.30))

        return round(
            0.65 * score_component + 0.35 * margin_component,
            3,
        )

    @staticmethod
    def _fallback(reason: str) -> IntentResult:
        return IntentResult(
            intent=Intent.DOCUMENT_QA,
            confidence=0.0,
            reason=reason,
            ambiguous=True,
        )
