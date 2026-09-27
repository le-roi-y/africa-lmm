from src.data.schemas import DocumentChunk, RetrievalResult
from src.models.base import BaseModel
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import ScoreReranker
from src.rag.retriever import Retriever


class FakeRetriever(Retriever):
    def __init__(self):
        self.chunks = [
            DocumentChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="La production agricole est de 100 tonnes.",
                page_number=5,
            ),
            DocumentChunk(
                chunk_id="chunk-2",
                document_id="doc-1",
                text="Le rapport concerne l'agriculture.",
                page_number=6,
            ),
        ]

    def index(self, chunks):
        self.chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5):
        results = [
            RetrievalResult(
                chunk=chunk,
                score=1.0 - index * 0.1,
                rank=index + 1,
            )
            for index, chunk in enumerate(self.chunks[:top_k])
        ]

        return results


class FakeModel(BaseModel):
    def __init__(self):
        super().__init__("fake-model")
        self.last_prompt = None

    def load(self):
        self.loaded = True

    def generate(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        return "La production agricole est de 100 tonnes."

    def unload(self):
        self.loaded = False


def test_rag_pipeline():
    retriever = FakeRetriever()
    model = FakeModel()

    pipeline = RAGPipeline(
        retriever=retriever,
        model=model,
        reranker=ScoreReranker(),
    )

    result = pipeline.answer(
        "Quelle est la production agricole ?",
        top_k=2,
    )

    assert result.answer == "La production agricole est de 100 tonnes."
    assert len(result.citations) == 2
    assert result.citations[0].document_id == "doc-1"
    assert result.citations[0].page_number == 5
    assert model.last_prompt is not None
    assert "Quelle est la production agricole ?" in model.last_prompt


def test_rag_pipeline_rejects_empty_question():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        model=FakeModel(),
    )

    try:
        pipeline.answer("")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
