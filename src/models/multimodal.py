from __future__ import annotations

from typing import Any

from .base import BaseModel


class MultimodalModel(BaseModel):
    """Interface pour les modèles texte + vision."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        raise NotImplementedError("Multimodal model backend is not connected yet.")

    def unload(self) -> None:
        self.loaded = False
