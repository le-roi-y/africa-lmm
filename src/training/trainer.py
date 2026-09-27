from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class TrainingConfig(BaseModel):

    output_dir: str = "outputs"
    epochs: int = 1
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 42


class Trainer(ABC):

    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:

        self.config = config

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError
