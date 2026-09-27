from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )


class LoRATrainer:
    def __init__(self, config: LoRAConfig | None = None) -> None:
        self.config = config or LoRAConfig()

    def build_config(self) -> Any:
        from peft import LoraConfig

        return LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=self.config.target_modules,
            task_type="CAUSAL_LM",
        )
