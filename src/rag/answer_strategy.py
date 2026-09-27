from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.data.schemas import RetrievalResult
from src.models.base import BaseModel
from src.rag.context_builder import ContextBuilder
from src.rag.extractive_qa import ExtractiveQA
from src.rag.table_qa import TableQA


class AnswerStrategy:
    """Selects the appropriate answer mechanism for retrieved context."""

    def __init__(
        self,
        model: BaseModel,
        reranker_model: CrossEncoder | None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.model = model
        self.context_builder = context_builder or ContextBuilder()
        self.table_qa = TableQA()
        self.extractive_qa = (
            ExtractiveQA(model=reranker_model)
            if reranker_model is not None
            else None
        )

    def summarize(
        self,
        results: list[RetrievalResult],
    ) -> str:
        """Generate a concise summary strictly from document context."""

        if not results:
            return (
                "Impossible de résumer le document : "
                "aucun contenu exploitable n'a été trouvé."
            )

        context = self.context_builder.build(results)

        prompt = (
            "You are AFRICA-LMM, a strict document summarization system.\n\n"
            "Summarize ONLY the provided document context.\n"
            "Never use outside knowledge.\n"
            "Never invent information.\n"
            "Identify the main ideas, important facts, figures and conclusions.\n"
            "Write the summary in French.\n"
            "Use clear bullet points when appropriate.\n"
            "Keep the summary concise but informative.\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            "SUMMARY:"
        )

        return self.model.generate(prompt)


    def compare(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """Compare documents strictly from their indexed context."""

        if not results:
            return (
                "Impossible de comparer les documents : "
                "aucun contenu exploitable n'a été trouvé."
            )

        context = self.context_builder.build(results)

        prompt = (
            "You are AFRICA-LMM, a strict document comparison system.\n\n"
            "Compare ONLY the information contained in the provided document context.\n"
            "Never use outside knowledge.\n"
            "Never invent missing information.\n"
            "Clearly distinguish the documents or sources being compared.\n"
            "Identify similarities, differences, trends, figures and important conclusions.\n"
            "Preserve names, numbers, units and facts exactly.\n"
            "Write the response in French.\n"
            "Use a structured comparison when useful.\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"USER REQUEST:\n{question}\n\n"
            "COMPARISON:"
        )

        return self.model.generate(prompt)

    def overview(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """Generate a broad overview strictly from document context."""

        if not results:
            return (
                "Impossible de présenter une vue d'ensemble : "
                "aucun contenu exploitable n'a été trouvé."
            )

        context = self.context_builder.build(results)

        prompt = (
            "You are AFRICA-LMM, a strict document overview system.\n\n"
            "Provide a broad understanding of the provided document context.\n"
            "Use ONLY the supplied context.\n"
            "Never use outside knowledge.\n"
            "Never invent information.\n"
            "Explain the subject, structure, major themes, important facts "
            "and conclusions when they are supported by the context.\n"
            "Write in French.\n"
            "Be informative but avoid unnecessary repetition.\n\n"
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"USER REQUEST:\n{question}\n\n"
            "OVERVIEW:"
        )

        return self.model.generate(prompt)

    def answer(
        self,
        question: str,
        results: list[RetrievalResult],
        contextual_question: str | None = None,
    ) -> str:
        if not results:
            return (
                "L'information demandée n'est pas disponible "
                "dans les documents fournis."
            )

        table_answer = self._table_answer(
            question=question,
            contextual_question=contextual_question,
            results=results,
        )

        if table_answer is not None:
            return table_answer

        if self.extractive_qa is not None:
            extractive_answer = self.extractive_qa.answer(
                question=question,
                results=results,
            )

            if extractive_answer is not None:
                return extractive_answer

        context = self.context_builder.build(results)

        prompt = (
            "You are AFRICA-LMM, a strict document question-answering system.\n\n"
            "Answer ONLY using the provided context.\n"
            "Never use outside knowledge.\n"
            "Never invent or estimate missing information.\n"
            "Answer directly and concisely.\n"
            "Preserve names, numbers, units and facts exactly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        return self.model.generate(prompt)

    def _table_answer(
        self,
        question: str,
        contextual_question: str | None,
        results: list[RetrievalResult],
    ) -> str | None:
        tables = []

        for result in results:
            tables.extend(
                result.chunk.metadata.get("tables", [])
            )

        return self.table_qa.answer(
            question=question,
            column_question=contextual_question,
            tables=tables,
        )
