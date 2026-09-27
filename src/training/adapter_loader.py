from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRAAdapterLoader:
    """Load a base causal LM with an AFRICA-LMM LoRA adapter."""

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str | Path,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path)

        if not self.adapter_path.exists():
            raise FileNotFoundError(self.adapter_path)

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
        )

        model = PeftModel.from_pretrained(
            model,
            self.adapter_path,
        )

        model.eval()

        return model, tokenizer
