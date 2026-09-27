from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


@dataclass
class SFTTrainingConfig:
    """Configuration du Supervised Fine-Tuning."""

    model_name: str
    output_dir: str = "outputs/sft"

    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    max_length: int = 2048
    seed: int = 42

    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be greater than 0.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                "per_device_train_batch_size must be greater than 0."
            )

        if self.per_device_eval_batch_size <= 0:
            raise ValueError(
                "per_device_eval_batch_size must be greater than 0."
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than 0."
            )

        if self.max_length <= 0:
            raise ValueError("max_length must be greater than 0.")

        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be greater than 0.")


class SFTPipeline:
    """
    Pipeline de Supervised Fine-Tuning pour AFRICA-LMM.

    Dataset
        ↓
    Tokenizer
        ↓
    SFTTrainer
        ↓
    LoRA
        ↓
    Adapter
    """

    def __init__(
        self,
        config: SFTTrainingConfig,
        lora_config: LoraConfig | None = None,
    ) -> None:
        self.config = config
        self.lora_config = lora_config

        self.tokenizer = None
        self.trainer: SFTTrainer | None = None

    def load_tokenizer(self) -> None:
        """Load the tokenizer only when explicitly requested."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build(
        self,
        dataset: DatasetDict,
    ) -> SFTTrainer:
        """Build the Hugging Face SFT trainer."""
        if "train" not in dataset:
            raise ValueError("Dataset must contain a 'train' split.")

        if "validation" not in dataset:
            raise ValueError(
                "Dataset must contain a 'validation' split."
            )

        if self.tokenizer is None:
            self.load_tokenizer()

        use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        use_fp16 = torch.cuda.is_available() and not use_bf16

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=(
                self.config.per_device_train_batch_size
            ),
            per_device_eval_batch_size=(
                self.config.per_device_eval_batch_size
            ),
            gradient_accumulation_steps=(
                self.config.gradient_accumulation_steps
            ),
            max_length=self.config.max_length,
            seed=self.config.seed,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            report_to="none",
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

        self.trainer = SFTTrainer(
            model=self.config.model_name,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=self.tokenizer,
            peft_config=self.lora_config,
        )

        return self.trainer

    def train(self) -> Any:
        """Run the training process."""
        if self.trainer is None:
            raise RuntimeError(
                "Trainer has not been built. Call build() first."
            )

        return self.trainer.train()

    def evaluate(self) -> dict[str, float]:
        """Evaluate the current model."""
        if self.trainer is None:
            raise RuntimeError(
                "Trainer has not been built. Call build() first."
            )

        metrics = self.trainer.evaluate()

        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

    def save(self, path: str | Path | None = None) -> None:
        """Save the trained model or LoRA adapter."""
        if self.trainer is None:
            raise RuntimeError(
                "Trainer has not been built. Call build() first."
            )

        output_path = Path(path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.trainer.save_model(str(output_path))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(output_path))
