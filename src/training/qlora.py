from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from .lora import LoRAConfig


class QLoRAConfig(BaseModel):
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization_bits: int = 4

    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class QLoRATrainer:
    """
    Configuration et préparation d'un entraînement QLoRA.

    QLoRA combine :
    - quantification 4-bit du modèle de base ;
    - adaptation LoRA ;
    - entraînement des paramètres LoRA uniquement.
    """

    def __init__(self, config: QLoRAConfig | None = None) -> None:
        self.config = config or QLoRAConfig()

    def build_lora_config(self) -> Any:
        from peft import LoraConfig

        return LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            task_type="CAUSAL_LM",
        )

    def build_quantization_config(self) -> Any:
        if self.config.quantization_bits != 4:
            raise ValueError("Currently only 4-bit QLoRA is supported.")

        from transformers import BitsAndBytesConfig

        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.config.bnb_4bit_compute_dtype)

        if compute_dtype is None:
            raise ValueError("Unsupported compute dtype: " f"{self.config.bnb_4bit_compute_dtype}")

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
