from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.data.schemas import Answer, RetrievalResult
from src.inference.rag_service import RAGService


BENCHMARK_PATH = Path(__file__).with_name("rag_eval.json")


class RAGEvaluator:
    """Évalue le retrieval, le reranking, les réponses et les citations."""

    def __init__(self, service: RAGService) -> None:
        self.service = service

    def load_dataset(self) -> list[dict[str, Any]]:
        with BENCHMARK_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def evaluate(self) -> dict[str, Any]:
        dataset = self.load_dataset()

        results: list[dict[str, Any]] = []

        retrieval_hits_at_1 = 0
        retrieval_hits_at_5 = 0
        reranker_hits_at_1 = 0
        answer_hits = 0
        citation_hits = 0
        unanswerable_hits = 0

        for item in dataset:
            question = item["question"]
            expected_chunk = item["expected_chunk"]
            expected_answer = item["expected_answer"]
            expected_document = item["expected_document"]
            expected_page = item["expected_page"]

            retrieved = self.service.retriever.retrieve(
                query=question,
                top_k=5,
            )

            retrieval_rank = self._find_chunk_rank(
                retrieved,
                expected_chunk,
            )

            reranked = self.service.reranker.rerank(
                query=question,
                results=retrieved,
            )

            reranker_rank = self._find_chunk_rank(
                reranked,
                expected_chunk,
            )

            answer = self.service.query(
                question=question,
                top_k=5,
            )

            answer_correct = self._answer_matches(
                answer,
                expected_answer,
                item["type"],
            )

            citation_correct = self._citation_matches(
                answer,
                expected_document,
                expected_page,
                expected_chunk,
            )

            unanswerable_correct = False

            if item["type"] == "unanswerable":
                unanswerable_correct = self._is_unanswerable_response(
                    answer
                )

            if retrieval_rank == 1:
                retrieval_hits_at_1 += 1

            if retrieval_rank is not None:
                retrieval_hits_at_5 += 1

            if reranker_rank == 1:
                reranker_hits_at_1 += 1

            if answer_correct:
                answer_hits += 1

            if citation_correct:
                citation_hits += 1

            if unanswerable_correct:
                unanswerable_hits += 1

            results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "type": item["type"],
                    "retrieval_rank": retrieval_rank,
                    "reranker_rank": reranker_rank,
                    "answer": answer.answer,
                    "expected_answer": expected_answer,
                    "answer_correct": answer_correct,
                    "citation_correct": citation_correct,
                    "unanswerable_correct": (
                        unanswerable_correct
                        if item["type"] == "unanswerable"
                        else None
                    ),
                }
            )

        total = len(dataset)
        unanswerable_total = sum(
            1
            for item in dataset
            if item["type"] == "unanswerable"
        )

        return {
            "dataset_size": total,
            "metrics": {
                "retrieval_recall_at_1": (
                    retrieval_hits_at_1 / total
                    if total
                    else 0.0
                ),
                "retrieval_recall_at_5": (
                    retrieval_hits_at_5 / total
                    if total
                    else 0.0
                ),
                "reranker_recall_at_1": (
                    reranker_hits_at_1 / total
                    if total
                    else 0.0
                ),
                "answer_accuracy": (
                    answer_hits / total
                    if total
                    else 0.0
                ),
                "citation_accuracy": (
                    citation_hits / total
                    if total
                    else 0.0
                ),
                "unanswerable_accuracy": (
                    unanswerable_hits / unanswerable_total
                    if unanswerable_total
                    else 0.0
                ),
            },
            "results": results,
        }

    @staticmethod
    def _find_chunk_rank(
        results: list[RetrievalResult],
        expected_chunk: str | None,
    ) -> int | None:
        if expected_chunk is None:
            return None

        for result in results:
            if result.chunk.chunk_id == expected_chunk:
                return result.rank

        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

    @classmethod
    def _answer_matches(
        cls,
        answer: Answer,
        expected_answer: str | None,
        question_type: str,
    ) -> bool:
        if question_type == "unanswerable":
            return False

        if expected_answer is None:
            return False

        actual = cls._normalize(answer.answer)
        expected = cls._normalize(expected_answer)

        return (
            actual == expected
            or expected in actual
        )

    @staticmethod
    def _citation_matches(
        answer: Answer,
        expected_document: str | None,
        expected_page: int | None,
        expected_chunk: str | None,
    ) -> bool:
        if expected_document is None:
            return False

        for citation in answer.citations:
            if citation.document_id != expected_document:
                continue

            if (
                expected_page is not None
                and citation.page_number != expected_page
            ):
                continue

            if (
                expected_chunk is not None
                and citation.chunk_id != expected_chunk
            ):
                continue

            return True

        return False

    @classmethod
    def _is_unanswerable_response(
        cls,
        answer: Answer,
    ) -> bool:
        text = cls._normalize(answer.answer)

        refusal_patterns = [
            "not available",
            "information is not available",
            "information n'est pas disponible",
            "n'est pas disponible",
            "pas disponible",
            "non disponible",
            "je ne dispose pas",
            "je ne peux pas",
            "aucune information",
            "inconnu",
        ]

        return any(
            pattern in text
            for pattern in refusal_patterns
        )


def main() -> None:
    service = RAGService(
        model_name="Qwen/Qwen2.5-0.5B-Instruct"
    )

    evaluator = RAGEvaluator(service)
    report = evaluator.evaluate()

    output_path = Path("tests/benchmarks/rag_eval_results.json")

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = report["metrics"]

    print("\n=== AFRICA-LMM RAG EVALUATION ===")
    print(f"Dataset size: {report['dataset_size']}")
    print(
        f"Retrieval Recall@1: "
        f"{metrics['retrieval_recall_at_1']:.2%}"
    )
    print(
        f"Retrieval Recall@5: "
        f"{metrics['retrieval_recall_at_5']:.2%}"
    )
    print(
        f"Reranker Recall@1: "
        f"{metrics['reranker_recall_at_1']:.2%}"
    )
    print(
        f"Answer Accuracy: "
        f"{metrics['answer_accuracy']:.2%}"
    )
    print(
        f"Citation Accuracy: "
        f"{metrics['citation_accuracy']:.2%}"
    )
    print(
        f"Unanswerable Accuracy: "
        f"{metrics['unanswerable_accuracy']:.2%}"
    )
    print(f"\nReport: {output_path}")


if __name__ == "__main__":
    main()
