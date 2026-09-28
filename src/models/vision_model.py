from __future__ import annotations

from typing import Any

from .base import BaseModel


class VisionModel(BaseModel):
    """Interface pour les modèles capables de comprendre des images."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        raise NotImplementedError("Vision model backend is not connected yet.")

    def unload(self) -> None:
        self.loaded = False
