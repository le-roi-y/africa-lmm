from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import mlflow

from src.inference.rag_service import RAGService

from .rag_evaluator import RAGEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "configs" / "datasets" / "rag_eval.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_results.json"

EXPERIMENT_NAME = "africa-lmm-rag-evaluation"


def main() -> None:
    print("\n=== AFRICA-LMM RAG BENCHMARK ===\n")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / "mlflow.db"}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    start = perf_counter()

    service = RAGService(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    evaluator = RAGEvaluator(
        service=service,
        dataset_path=DATASET_PATH,
    )

    with mlflow.start_run(run_name="rag-benchmark") as run:
        report = evaluator.evaluate()
        duration = perf_counter() - start

        metrics = report["metrics"]

        mlflow.log_params(
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "reranker": "BAAI/bge-reranker-v2-m3",
                "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
                "dataset_size": report["dataset_size"],
            }
        )

        mlflow.log_metrics(
            {
                "retrieval_recall_at_1": metrics["retrieval_recall_at_1"],
                "retrieval_recall_at_5": metrics["retrieval_recall_at_5"],
                "reranker_recall_at_1": metrics["reranker_recall_at_1"],
                "answer_accuracy": metrics["answer_accuracy"],
                "citation_accuracy": metrics["citation_accuracy"],
                "unanswerable_accuracy": metrics["unanswerable_accuracy"],
                "duration_seconds": duration,
            }
        )

        RESULTS_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(RESULTS_PATH),
            artifact_path="evaluation",
        )

        mlflow.set_tags(
            {
                "project": "AFRICA-LMM",
                "task": "RAG evaluation",
                "framework": "PyTorch + Transformers",
            }
        )

        print(f"MLflow Run ID: {run.info.run_id}")

    print(f"\nDataset size: {report['dataset_size']}")

    print(f"\nRetrieval Recall@1: " f"{metrics['retrieval_recall_at_1']:.2%}")
    print(f"Retrieval Recall@5: " f"{metrics['retrieval_recall_at_5']:.2%}")
    print(f"Reranker Recall@1: " f"{metrics['reranker_recall_at_1']:.2%}")
    print(f"Answer Accuracy: {metrics['answer_accuracy']:.2%}")
    print(f"Citation Accuracy: {metrics['citation_accuracy']:.2%}")
    print(f"Unanswerable Accuracy: " f"{metrics['unanswerable_accuracy']:.2%}")

    print(f"\nDuration: {duration:.2f}s")
    print(f"Report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
