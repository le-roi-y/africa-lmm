from src.data.schemas import DocumentChunk, RetrievalResult
from src.rag.reranker import ScoreReranker


def make_result(chunk_id: str, score: float, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"Text {chunk_id}",
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        rank=rank,
    )


def test_score_reranker_sorts_results():
    results = [
        make_result("chunk-a", 0.30, 1),
        make_result("chunk-b", 0.95, 2),
        make_result("chunk-c", 0.60, 3),
    ]

    ranked = ScoreReranker().rerank("question", results)

    assert [result.chunk.chunk_id for result in ranked] == [
        "chunk-b",
        "chunk-c",
        "chunk-a",
    ]

    assert [result.rank for result in ranked] == [1, 2, 3]
