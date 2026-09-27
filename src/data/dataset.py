from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


class DatasetFormatError(ValueError):
    """Raised when a training dataset has an invalid format."""


class InstructionDatasetBuilder:
    """
    Build Hugging Face datasets for instruction tuning.

    Supported input formats:
    - JSON
    - JSONL

    Required fields:
    - instruction
    - response

    Optional fields:
    - input
    - language
    - domain
    - country
    - source

    Output fields:
    - prompt
    - completion
    - instruction
    - input
    - language
    - domain
    - country
    - source

    The prompt/completion format is compatible with
    Hugging Face TRL SFTTrainer.
    """

    REQUIRED_FIELDS = {"instruction", "response"}

    def __init__(
        self,
        validation_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not 0 < validation_split < 1:
            raise ValueError(
                "validation_split must be between 0 and 1."
            )

        self.validation_split = validation_split
        self.seed = seed

    def load_file(
        self,
        path: str | Path,
    ) -> list[dict[str, Any]]:
        """Load examples from a JSON or JSONL file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._load_json(path)

        if suffix == ".jsonl":
            return self._load_jsonl(path)

        raise ValueError(
            f"Unsupported dataset format: {suffix}. "
            "Expected .json or .jsonl."
        )

    def build(
        self,
        path: str | Path,
    ) -> DatasetDict:
        """
        Load, validate, format, and split the dataset.
        """
        examples = self.load_file(path)
        validated = self._validate_examples(examples)

        formatted = [
            self._format_example(example)
            for example in validated
        ]

        dataset = Dataset.from_list(formatted)

        split = dataset.train_test_split(
            test_size=self.validation_split,
            seed=self.seed,
        )

        return DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )

    def _load_json(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        """Load examples from a JSON file."""
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            data = data.get("data")

        if not isinstance(data, list):
            raise DatasetFormatError(
                "JSON dataset must contain a list of examples."
            )

        return data

    def _load_jsonl(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        """Load examples from a JSONL file."""
        examples: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        f"Invalid JSON on line {line_number}."
                    ) from exc

                examples.append(example)

        return examples

    def _validate_examples(
        self,
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate the dataset structure."""
        if not examples:
            raise DatasetFormatError(
                "Dataset contains no examples."
            )

        validated: list[dict[str, Any]] = []

        for index, example in enumerate(examples):
            if not isinstance(example, dict):
                raise DatasetFormatError(
                    f"Example {index} must be a JSON object."
                )

            missing = self.REQUIRED_FIELDS - example.keys()

            if missing:
                raise DatasetFormatError(
                    f"Example {index} is missing required fields: "
                    f"{sorted(missing)}"
                )

            instruction = example["instruction"]
            response = example["response"]

            if (
                not isinstance(instruction, str)
                or not instruction.strip()
            ):
                raise DatasetFormatError(
                    f"Example {index} has an invalid instruction."
                )

            if (
                not isinstance(response, str)
                or not response.strip()
            ):
                raise DatasetFormatError(
                    f"Example {index} has an invalid response."
                )

            validated.append(example)

        return validated

    @staticmethod
    def _format_example(
        example: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert a raw example to the TRL SFT format.

        The prompt contains the instruction and optional input.

        The completion contains the expected assistant response.
        """
        instruction = example["instruction"].strip()
        response = example["response"].strip()
        user_input = str(
            example.get("input", "")
        ).strip()

        if user_input:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{user_input}\n\n"
                "### Response:"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                "### Response:"
            )

        return {
            "prompt": prompt,
            "completion": response,
            "instruction": instruction,
            "input": user_input,
            "language": example.get("language"),
            "domain": example.get("domain"),
            "country": example.get("country"),
            "source": example.get("source"),
        }
