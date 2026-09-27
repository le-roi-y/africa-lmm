from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig

from src.data.dataset import InstructionDatasetBuilder
from src.training.sft import SFTPipeline, SFTTrainingConfig


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AFRICA-LMM with supervised fine-tuning."
    )

    parser.add_argument(
        "--config",
        default="configs/experiments/sft.yaml",
        help="Path to the SFT YAML configuration.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Override the Hugging Face model name.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the training dataset path.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override the number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the learning rate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-device batch size.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override the maximum sequence length.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually start training.",
    )

    return parser.parse_args()


def get_value(
    override: Any,
    config: dict[str, Any],
    section: str,
    key: str,
) -> Any:
    if override is not None:
        return override

    try:
        return config[section][key]
    except KeyError as exc:
        raise ValueError(
            f"Missing configuration value: {section}.{key}"
        ) from exc


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = get_value(
        args.model,
        config,
        "model",
        "name",
    )

    dataset_path = Path(
        get_value(
            args.dataset,
            config,
            "dataset",
            "path",
        )
    )

    output_dir = get_value(
        args.output_dir,
        config,
        "training",
        "output_dir",
    )

    epochs = get_value(
        args.epochs,
        config,
        "training",
        "num_train_epochs",
    )

    learning_rate = get_value(
        args.learning_rate,
        config,
        "training",
        "learning_rate",
    )

    batch_size = get_value(
        args.batch_size,
        config,
        "training",
        "per_device_train_batch_size",
    )

    max_length = get_value(
        args.max_length,
        config,
        "training",
        "max_length",
    )

    validation_split = config["dataset"]["validation_split"]
    seed = config["dataset"]["seed"]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    lora = config["lora"]

    lora_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )

    training_config = SFTTrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config["training"][
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=config["training"][
            "gradient_accumulation_steps"
        ],
        max_length=max_length,
        seed=seed,
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy=config["training"]["eval_strategy"],
        gradient_checkpointing=config["training"][
            "gradient_checkpointing"
        ],
    )

    print("=" * 60)
    print("AFRICA-LMM — Supervised Fine-Tuning")
    print("=" * 60)
    print(f"Config      : {args.config}")
    print(f"Model       : {model_name}")
    print(f"Dataset     : {dataset_path}")
    print(f"Output      : {output_dir}")
    print(f"Epochs      : {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Max length  : {max_length}")
    print(f"LoRA rank   : {lora['rank']}")
    print(f"LoRA alpha   : {lora['alpha']}")
    print()

    print("Loading dataset...")

    dataset_builder = InstructionDatasetBuilder(
        validation_split=validation_split,
        seed=seed,
    )

    dataset = dataset_builder.build(dataset_path)

    print(f"Train examples      : {len(dataset['train'])}")
    print(f"Validation examples : {len(dataset['validation'])}")
    print()

    pipeline = SFTPipeline(
        config=training_config,
        lora_config=lora_config,
    )

    print("Building SFT trainer...")
    pipeline.build(dataset)

    print()
    print("SFT trainer: READY")

    if not args.train:
        print()
        print(
            "Training not started. "
            "Use --train to launch the actual training."
        )
        return

    print()
    print("Starting training...")
    print()

    pipeline.train()

    print()
    print("Evaluating...")

    metrics = pipeline.evaluate()

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print()
    print("Saving model...")

    pipeline.save()

    print()
    print("=" * 60)
    print("Training completed.")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
