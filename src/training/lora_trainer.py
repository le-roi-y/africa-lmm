from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


class LoRATrainer:
    """Fine-tuning LoRA pour AFRICA-LMM."""

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
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )

        dataset = load_dataset(
            "json",
            data_files=self.dataset_path,
            split="train",
        )

        def format_example(example: dict[str, Any]) -> str:
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]

            user_content = instruction

            if input_text.strip():
                user_content += f"\n\nContexte:\n{input_text}"

            messages = [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": output,
                },
            ]

            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        dataset = dataset.map(
            lambda example: {
                "text": format_example(example),
            }
        )

        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["alpha"],
            lora_dropout=self.config["lora"]["dropout"],
            target_modules=self.config["lora"]["target_modules"],
            task_type="CAUSAL_LM",
            bias="none",
        )

        training_config = self.config["training"]

        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=training_config["learning_rate"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=training_config["save_total_limit"],
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(training_config["output_dir"])
        tokenizer.save_pretrained(training_config["output_dir"])
