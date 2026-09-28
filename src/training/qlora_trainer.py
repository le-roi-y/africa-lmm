from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


class QLoRATrainer:
    """Memory-efficient 4-bit QLoRA fine-tuning for AFRICA-LMM."""

    def __init__(self, config_path: str | Path) -> None:
        self.config = self._load_config(config_path)
        self.model_name = self.config["model"]["name"]
        self.dataset_path = self.config["dataset"]["path"]

    @staticmethod
    def _load_config(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Training configuration must be a mapping.")

        return config

    def train(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization = self.config["quantization"]

        compute_dtype = getattr(
            torch,
            quantization["compute_dtype"],
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization["load_in_4bit"],
            bnb_4bit_quant_type=quantization["quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quantization["double_quant"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        dataset = load_dataset(
            "json",
            data_files=self.dataset_path,
            split="train",
        )

        def format_example(example: dict[str, Any]) -> dict[str, str]:
            messages = [
                {
                    "role": "user",
                    "content": example["instruction"],
                },
                {
                    "role": "assistant",
                    "content": example["output"],
                },
            ]

            return {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }

        dataset = dataset.map(format_example)

        lora = self.config["lora"]

        peft_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
            bias="none",
        )

        training = self.config["training"]

        training_args = TrainingArguments(
            output_dir=training["output_dir"],
            num_train_epochs=training["num_train_epochs"],
            per_device_train_batch_size=training["per_device_train_batch_size"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            learning_rate=training["learning_rate"],
            logging_steps=training["logging_steps"],
            save_steps=training["save_steps"],
            save_total_limit=training["save_total_limit"],
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()

        trainer.save_model(training["output_dir"])
        tokenizer.save_pretrained(training["output_dir"])
