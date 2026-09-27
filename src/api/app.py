from __future__ import annotations

from fastapi import FastAPI

from src.inference.predictor import Predictor

from .routes import APIRoutes
from .schemas import QueryRequest, QueryResponse


def create_app(predictor: Predictor) -> FastAPI:
    """Create and configure the AFRICA-LMM FastAPI application."""

    app = FastAPI(
        title="AFRICA-LMM",
        description="Multimodal AI platform for African documents.",
        version="0.1.0",
    )

    routes = APIRoutes(predictor)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health-check endpoint."""
        return {
            "status": "ok",
            "service": "africa-lmm",
        }

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        """Answer a question using the AFRICA-LMM RAG pipeline."""
        return routes.query(request)

    return app
