from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationResolution:
    question: str
    resolved_question: str
    context: str | None = None
    entities: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceExtraction:
    entities: tuple[str, ...]
    years: tuple[str, ...]


class ConversationResolver:
    """
    Résout les références implicites d'une question à partir
    des questions précédentes de l'utilisateur.

    Le resolver reste indépendant du domaine :
    aucun produit, pays, dataset ou vocabulaire métier
    n'est codé en dur.
    """

    YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    ENTITY_PATTERNS = (
        re.compile(
            r"\b(?:pour|concernant|regarding|about|for)\s+"
            r"(?:le|la|les|l'|un|une|the|a|an)\s+"
            r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]*(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]*){0,3})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:de|du|des|d')\s+"
            r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]*)",
            re.IGNORECASE,
        ),
    )

    FOLLOW_UP_MARKERS = (
        "et ",
        "pour ",
        "en ",
        "qu'en ",
        "qu’en ",
        "concernant ",
        "regarding ",
        "about ",
        "for ",
        "ce ",
        "cette ",
        "cet ",
        "celui ",
        "celle ",
        "ce produit",
        "cette année",
        "l'année précédente",
        "l’année précédente",
        "l'année dernière",
        "l’année dernière",
    )

    GENERIC_REFERENCES = {
        "ce",
        "cet",
        "cette",
        "ces",
        "celui",
        "celle",
        "ceux",
        "celles",
        "produit",
        "produits",
        "année",
        "annee",
        "document",
        "documents",
    }

    def resolve(
        self,
        question: str,
        conversation_context: str | None = None,
        document_ids: list[str] | None = None,
    ) -> ConversationResolution:
        question = question.strip()
        selected_documents = tuple(document_ids or [])
        current_refs = self.extract_references(question)

        if not conversation_context:
            return ConversationResolution(
                question=question,
                resolved_question=question,
                entities=current_refs.entities,
                years=current_refs.years,
                document_ids=selected_documents,
            )

        previous_questions = [
            line.strip()
            for line in conversation_context.splitlines()
            if line.strip()
        ]

        if not previous_questions:
            return ConversationResolution(
                question=question,
                resolved_question=question,
                entities=current_refs.entities,
                years=current_refs.years,
                document_ids=selected_documents,
            )

        # Reconstruct the conversation state progressively.
        resolved_previous = previous_questions[0]

        for previous_question in previous_questions[1:]:
            previous_refs = self.extract_references(resolved_previous)
            current_previous_refs = self.extract_references(previous_question)

            resolved_previous = self._resolve_question(
                question=previous_question,
                current_refs=current_previous_refs,
                previous_question=resolved_previous,
                previous_refs=previous_refs,
            )

        previous_refs = self.extract_references(resolved_previous)

        resolved_question = self._resolve_question(
            question=question,
            current_refs=current_refs,
            previous_question=resolved_previous,
            previous_refs=previous_refs,
        )

        resolved_refs = self.extract_references(resolved_question)

        return ConversationResolution(
            question=question,
            resolved_question=resolved_question,
            context="\n".join(previous_questions[-3:]),
            entities=resolved_refs.entities,
            years=resolved_refs.years,
            document_ids=selected_documents,
        )

    def extract_references(self, text: str) -> ReferenceExtraction:
        years = tuple(dict.fromkeys(self.YEAR_PATTERN.findall(text)))

        entities: list[str] = []

        for pattern in self.ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                candidate = self._clean_entity(match.group(1))

                if not candidate:
                    continue

                if candidate.lower() in self.GENERIC_REFERENCES:
                    continue

                if self.YEAR_PATTERN.fullmatch(candidate):
                    continue

                if candidate not in entities:
                    entities.append(candidate)

        return ReferenceExtraction(
            entities=tuple(entities),
            years=years,
        )

    def _resolve_question(
        self,
        question: str,
        current_refs: ReferenceExtraction,
        previous_question: str,
        previous_refs: ReferenceExtraction,
    ) -> str:
        if not self._is_follow_up(question):
            return question

        resolved = question

        # Si la question actuelle ne donne pas d'entité,
        # on conserve l'entité de la question précédente.
        if not current_refs.entities and previous_refs.entities:
            entity = previous_refs.entities[0]

            if current_refs.years:
                resolved = self._replace_year_in_previous_question(
                    previous_question=previous_question,
                    current_year=current_refs.years[0],
                )
            else:
                resolved = (
                    f"{previous_question}\n\n"
                    f"Question actuelle : {question}"
                )

                resolved = (
                    f"Contexte résolu : {entity}\n"
                    f"Question actuelle : {question}"
                )

        # Si l'entité est nouvelle mais que l'année est implicite,
        # on hérite de l'année précédente.
        elif current_refs.entities and not current_refs.years:
            if previous_refs.years:
                resolved = self._replace_entity_in_previous_question(
                    previous_question=previous_question,
                    previous_entity=previous_refs.entities[0],
                    current_entity=current_refs.entities[0],
                )

        # Si l'entité et l'année sont déjà explicites,
        # la formulation actuelle suffit.
        elif current_refs.entities and current_refs.years:
            resolved = question

        # Fallback générique pour les références pronominales.
        else:
            resolved = (
                f"Contexte précédent : {previous_question}\n\n"
                f"Question actuelle : {question}"
            )

        return resolved

    def _replace_year_in_previous_question(
        self,
        previous_question: str,
        current_year: str,
    ) -> str:
        if self.YEAR_PATTERN.search(previous_question):
            return self.YEAR_PATTERN.sub(
                current_year,
                previous_question,
                count=1,
            )

        return (
            f"{previous_question.rstrip(' ?')} "
            f"pour l'année {current_year} ?"
        )

    def _replace_entity_in_previous_question(
        self,
        previous_question: str,
        previous_entity: str,
        current_entity: str,
    ) -> str:
        pattern = re.compile(
            re.escape(previous_entity),
            re.IGNORECASE,
        )

        if pattern.search(previous_question):
            return pattern.sub(
                current_entity,
                previous_question,
                count=1,
            )

        return (
            f"{previous_question.rstrip(' ?')} "
            f"pour {current_entity} ?"
        )

    def _is_follow_up(self, question: str) -> bool:
        normalized = " ".join(question.lower().split()).strip()

        if normalized.endswith("?"):
            normalized = normalized[:-1].strip()

        if not normalized:
            return False

        if normalized.startswith(self.FOLLOW_UP_MARKERS):
            return True

        words = normalized.split()

        if len(words) <= 5:
            return normalized.startswith(
                (
                    "et ",
                    "pour ",
                    "en ",
                    "qu'en ",
                    "qu’en ",
                    "quel ",
                    "quelle ",
                    "quels ",
                    "quelles ",
                )
            )

        return False

    @staticmethod
    def _clean_entity(value: str) -> str:
        value = " ".join(value.split()).strip(" ,.;:!?")

        # Évite de considérer une année comme une entité.
        if re.fullmatch(r"(?:19|20)\d{2}", value):
            return ""

        return value
