from __future__ import annotations

from src.data.schemas import Answer, Citation
from src.models.base import BaseModel

from .answer_strategy import AnswerStrategy
from .conversation_resolver import ConversationResolver
from .intent_router import Intent, IntentRouter, RouterContext
from .reranker import Reranker
from .retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        model: BaseModel,
        reranker: Reranker | None = None,
        answer_strategy: AnswerStrategy | None = None,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.reranker = reranker

        if answer_strategy is None:
            raise ValueError("RAGPipeline requires an AnswerStrategy.")

        self.answer_strategy = answer_strategy
        self.intent_router = IntentRouter(
            model=self.model,
            encoder_provider=self.retriever.get_encoder,
        )
        self.conversation_resolver = ConversationResolver()

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        conversation_context: str | None = None,
    ) -> Answer:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        selected_documents = set(document_ids or [])

        resolution = self.conversation_resolver.resolve(
            question=question,
            conversation_context=conversation_context,
            document_ids=document_ids,
        )

        retrieval_question = resolution.resolved_question

        intent_context = RouterContext(
            selected_document_count=len(selected_documents),
            selected_document_ids=tuple(sorted(selected_documents)),
        )

        intent = self.intent_router.route(
            question=retrieval_question,
            context=intent_context,
        )

        if intent.intent == Intent.DOCUMENT_SUMMARY and selected_documents:
            return self._summarize_documents(
                document_ids=selected_documents,
                question=question,
            )

        if intent.intent == Intent.DOCUMENT_COMPARISON and selected_documents:
            return self._compare_documents(
                document_ids=selected_documents,
                question=question,
            )

        if intent.intent == Intent.DOCUMENT_OVERVIEW and selected_documents:
            return self._overview_documents(
                document_ids=selected_documents,
                question=question,
            )

        # Conversation-aware multi-query retrieval.
        #
        # A short follow-up such as "Et pour le café ?" is often
        # semantically too weak on its own. We therefore retrieve
        # using both the current question and the previous questions,
        # then merge the candidates before reranking.
        retrieval_k = max(top_k, 25) if selected_documents else top_k

        retrieval_queries = [retrieval_question]

        result_map = {}

        for retrieval_query in retrieval_queries:
            query_results = self.retriever.retrieve(
                query=retrieval_query,
                top_k=retrieval_k,
            )

            for result in query_results:
                if selected_documents and result.chunk.document_id not in selected_documents:
                    continue

                result_map[result.chunk.chunk_id] = result

        results = list(result_map.values())

        if self.reranker is not None and results:
            results = self.reranker.rerank(
                query=retrieval_question,
                results=results,
            )

        results = results[:top_k]

        citations = [
            Citation(
                document_id=result.chunk.document_id,
                page_number=result.chunk.page_number,
                chunk_id=result.chunk.chunk_id,
            )
            for result in results
        ]

        if not results:
            return Answer(
                question=question,
                answer=(
                    "L'information demandée n'est pas disponible " "dans les documents fournis."
                ),
                citations=[],
                retrieved_chunks=[],
            )

        answer_text = self.answer_strategy.answer(
            question=question,
            contextual_question=retrieval_question,
            results=results,
        )

        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=results,
        )

    def _load_full_documents(
        self,
        document_ids: set[str],
    ) -> list:
        """Load indexed chunks for the selected documents."""

        retrieve_document = getattr(
            self.retriever,
            "retrieve_document",
            None,
        )

        if retrieve_document is None:
            return []

        results: list = []

        for document_id in document_ids:
            results.extend(
                retrieve_document(
                    document_id=document_id,
                    limit=100,
                )
            )

        results.sort(
            key=lambda result: (
                result.chunk.document_id,
                result.chunk.page_number or 0,
                result.chunk.chunk_id,
            )
        )

        return results

    def _compare_documents(
        self,
        document_ids: set[str],
        question: str,
    ) -> Answer:
        results = self._load_full_documents(document_ids)

        if not results:
            return Answer(
                question=question,
                answer=(
                    "Impossible de comparer les documents : " "aucun contenu indexé n'a été trouvé."
                ),
                citations=[],
                retrieved_chunks=[],
            )

        answer_text = self.answer_strategy.compare(
            question=question,
            results=results,
        )

        citations = [
            Citation(
                document_id=result.chunk.document_id,
                page_number=result.chunk.page_number,
                chunk_id=result.chunk.chunk_id,
            )
            for result in results
        ]

        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=results,
        )

    def _overview_documents(
        self,
        document_ids: set[str],
        question: str,
    ) -> Answer:
        results = self._load_full_documents(document_ids)

        if not results:
            return Answer(
                question=question,
                answer=(
                    "Impossible de présenter une vue d'ensemble : "
                    "aucun contenu indexé n'a été trouvé."
                ),
                citations=[],
                retrieved_chunks=[],
            )

        answer_text = self.answer_strategy.overview(
            question=question,
            results=results,
        )

        citations = [
            Citation(
                document_id=result.chunk.document_id,
                page_number=result.chunk.page_number,
                chunk_id=result.chunk.chunk_id,
            )
            for result in results
        ]

        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=results,
        )

    def _summarize_documents(
        self,
        document_ids: set[str],
        question: str,
    ) -> Answer:
        results = self._load_full_documents(document_ids)

        if not results:
            return Answer(
                question=question,
                answer=(
                    "Impossible de résumer le document : " "aucun contenu indexé n'a été trouvé."
                ),
                citations=[],
                retrieved_chunks=[],
            )

        answer_text = self.answer_strategy.summarize(results)

        citations = [
            Citation(
                document_id=result.chunk.document_id,
                page_number=result.chunk.page_number,
                chunk_id=result.chunk.chunk_id,
            )
            for result in results
        ]

        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=results,
        )
