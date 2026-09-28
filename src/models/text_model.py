from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import BaseModel


class TextModel(BaseModel):
    def __init__(self, model_name: str, device: str | None = None) -> None:
        super().__init__(model_name)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: Any = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
        )

        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def generate(
        self,
        prompt: str,
        **kwargs: object,
    ) -> str:
        if not self.loaded:
            raise RuntimeError("Model must be loaded before generation.")

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model resources are not initialized.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": 160,
            "do_sample": bool(kwargs.get("temperature", 0.0) > 0),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs.update(kwargs)

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
