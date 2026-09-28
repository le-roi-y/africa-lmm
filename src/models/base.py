from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError
