from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from src.models.multimodal import MultimodalModel
from src.models.text_model import TextModel
from src.models.vision_model import VisionModel


def test_text_model_initializes_with_explicit_device() -> None:
    model = TextModel("test-model", device="cpu")

    assert model.model_name == "test-model"
    assert model.device == "cpu"
    assert model.tokenizer is None
    assert model.model is None
    assert model.loaded is False


def test_text_model_loads_resources() -> None:
    tokenizer = Mock()
    model_backend = Mock()

    with (
        patch(
            "src.models.text_model.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as tokenizer_loader,
        patch(
            "src.models.text_model.AutoModelForCausalLM.from_pretrained",
            return_value=model_backend,
        ) as model_loader,
    ):
        model = TextModel("test-model", device="cpu")
        model.load()

    tokenizer_loader.assert_called_once_with("test-model")
    model_loader.assert_called_once_with(
        "test-model",
        torch_dtype=torch.float32,
    )

    model_backend.to.assert_called_once_with("cpu")
    model_backend.eval.assert_called_once_with()

    assert model.tokenizer is tokenizer
    assert model.model is model_backend
    assert model.loaded is True


def test_text_model_generate_requires_loaded_model() -> None:
    model = TextModel("test-model", device="cpu")

    with pytest.raises(
        RuntimeError,
        match="Model must be loaded before generation.",
    ):
        model.generate("Bonjour")


def test_text_model_generate_requires_initialized_resources() -> None:
    model = TextModel("test-model", device="cpu")
    model.loaded = True

    with pytest.raises(
        RuntimeError,
        match="Model resources are not initialized.",
    ):
        model.generate("Bonjour")


def test_text_model_generate_decodes_only_new_tokens() -> None:
    tokenizer = Mock()

    input_ids = torch.tensor([[10, 20, 30]])
    generated_ids = torch.tensor([[10, 20, 30, 40, 50]])

    tokenizer.return_value = {
        "input_ids": input_ids,
    }
    tokenizer.decode.return_value = "Bonjour le monde"

    model_backend = Mock()
    model_backend.generate.return_value = generated_ids

    model = TextModel("test-model", device="cpu")
    model.tokenizer = tokenizer
    model.model = model_backend
    model.loaded = True

    result = model.generate(
        "Bonjour",
        max_new_tokens=32,
        temperature=0.0,
        do_test=True,
    )

    assert result == "Bonjour le monde"

    tokenizer.assert_called_once_with(
        "Bonjour",
        return_tensors="pt",
    )

    model_backend.generate.assert_called_once()

    call_kwargs = model_backend.generate.call_args.kwargs

    assert call_kwargs["max_new_tokens"] == 32
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["do_sample"] is False
    assert call_kwargs["do_test"] is True

    tokenizer.decode.assert_called_once()

    decode_args, decode_kwargs = tokenizer.decode.call_args

    assert torch.equal(decode_args[0], generated_ids[0][3:])
    assert decode_kwargs == {
        "skip_special_tokens": True,
    }


def test_text_model_generate_enables_sampling_for_positive_temperature() -> None:
    tokenizer = Mock()

    input_ids = torch.tensor([[1, 2]])
    generated_ids = torch.tensor([[1, 2, 3]])

    tokenizer.return_value = {
        "input_ids": input_ids,
    }
    tokenizer.decode.return_value = "réponse"

    model_backend = Mock()
    model_backend.generate.return_value = generated_ids

    model = TextModel("test-model", device="cpu")
    model.tokenizer = tokenizer
    model.model = model_backend
    model.loaded = True

    result = model.generate(
        "Question",
        temperature=0.7,
    )

    assert result == "réponse"

    call_kwargs = model_backend.generate.call_args.kwargs

    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["do_sample"] is True


def test_text_model_unload_releases_resources() -> None:
    model = TextModel("test-model", device="cpu")
    model.tokenizer = Mock()
    model.model = Mock()
    model.loaded = True

    with patch("src.models.text_model.torch.cuda.is_available", return_value=False):
        model.unload()

    assert model.tokenizer is None
    assert model.model is None
    assert model.loaded is False


def test_multimodal_model_load_and_unload() -> None:
    model = MultimodalModel("test-multimodal")

    assert model.model_name == "test-multimodal"
    assert model.loaded is False

    model.load()

    assert model.loaded is True

    model.unload()

    assert model.loaded is False


def test_multimodal_model_requires_loading_before_generation() -> None:
    model = MultimodalModel("test-multimodal")

    with pytest.raises(
        RuntimeError,
        match="Model is not loaded.",
    ):
        model.generate("Describe this image.")


def test_multimodal_model_reports_unconnected_backend() -> None:
    model = MultimodalModel("test-multimodal")
    model.load()

    with pytest.raises(
        NotImplementedError,
        match="Multimodal model backend is not connected yet.",
    ):
        model.generate(
            "Describe this image.",
            images=["image.png"],
        )


def test_vision_model_load_and_unload() -> None:
    model = VisionModel("test-vision")

    assert model.model_name == "test-vision"
    assert model.loaded is False

    model.load()

    assert model.loaded is True

    model.unload()

    assert model.loaded is False


def test_vision_model_requires_loading_before_generation() -> None:
    model = VisionModel("test-vision")

    with pytest.raises(
        RuntimeError,
        match="Model is not loaded.",
    ):
        model.generate("Describe the image.")


def test_vision_model_reports_unconnected_backend() -> None:
    model = VisionModel("test-vision")
    model.load()

    with pytest.raises(
        NotImplementedError,
        match="Vision model backend is not connected yet.",
    ):
        model.generate("Describe the image.")
