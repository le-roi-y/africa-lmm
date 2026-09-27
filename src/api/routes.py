from __future__ import annotations

from src.inference.predictor import Predictor

from .schemas import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
)


class APIRoutes:

    def __init__(
        self,
        predictor: Predictor,
    ) -> None:

        self.predictor = predictor

    def query(
        self,
        request: QueryRequest,
    ) -> QueryResponse:

        result = self.predictor.predict(
            question=request.question,
            top_k=request.top_k,
        )

        citations = [
            CitationResponse(
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
            )
            for c in result.citations
        ]

        return QueryResponse(
            answer=result.answer,
            citations=citations,
        )
