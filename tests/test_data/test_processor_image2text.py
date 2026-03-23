"""Tests for Image2TextTranslationProcessor."""

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.legacy.image2text_preprocessor import (
    Image2TextTranslationProcessor,
)
from tests.test_data.conftest import FONT_PATH


def _modality_proc(processor):
    """Return the underlying ImageModalityProcessor from the wrapper."""
    return processor.slots[0].processor


class TestImageToTensor:
    def test_from_png_file(self, tokenizer, dummy_image_file):
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        tensor = _modality_proc(processor).process_sample(dummy_image_file)
        assert isinstance(tensor, torch.Tensor)

    def test_from_npy_file(self, tokenizer, tmp_path):
        """Loading from .npy image file."""
        path = str(tmp_path / "image.npy")
        np.save(path, np.random.rand(64, 64, 3).astype(np.float32))
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        tensor = _modality_proc(processor).process_sample(path)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (64, 64, 3)

    def test_from_ndarray(self, tokenizer):
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        arr = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = _modality_proc(processor).process_sample(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (64, 64, 3)

    def test_from_tensor(self, tokenizer):
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        t = torch.randn(3, 64, 64)
        result = _modality_proc(processor).process_sample(t)
        assert torch.equal(result, t)

    def test_from_text_string(self, tokenizer):
        """When signal is text (not a file path), it renders via font."""
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=224,
            height=224,
            normalize_image=False,
        )
        tensor = _modality_proc(processor).process_sample("Hello world")
        assert isinstance(tensor, torch.Tensor)
        # get_images returns (N_words, C, H, W) as float32
        assert tensor.ndim == 4
        assert tensor.shape[0] == 2  # "Hello" and "world"

    def test_normalization_applied(self, tokenizer, dummy_image_file):
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        tensor = _modality_proc(processor).process_sample(dummy_image_file)
        assert isinstance(tensor, torch.Tensor)

    def test_normalization_requires_mean_std(self, tokenizer):
        """Should raise error when normalize=True but mean/std not provided."""
        with pytest.raises(ValueError, match="mean"):
            Image2TextTranslationProcessor(
                tokenizer=tokenizer,
                normalize_image=True,
                mean=None,
                std=None,
            )


class TestImageObtainMultimodalInputAndMasks:
    def test_returns_input_frames_and_mask(self, tokenizer, dummy_image_file):
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        batch = [
            {
                "signal": dummy_image_file,
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "test",
            },
        ]
        result = processor(batch=batch)
        assert "input_frames" in result
        assert "attention_mask" in result


class TestImageTransformGetItemsOutput:
    def test_converts_signals_to_tensors(self, tokenizer, dummy_image_file):
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        batch = {
            "signal": [dummy_image_file],
        }
        result = processor._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)


class TestImageProcessorCall:
    """Full __call__() tests — the path exercised by DataCollatorMultimodalSeq2Seq."""

    EXPECTED_KEYS = {
        "input_frames",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    }

    def test_returns_batch_feature(self, tokenizer, image_batch_samples):
        """__call__ should return a BatchFeature (HF-compatible mapping)."""
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        result = processor(batch=image_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_has_all_expected_keys(self, tokenizer, image_batch_samples):
        """Output must contain all keys consumed by the model forward()."""
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        result = processor(batch=image_batch_samples)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_batch_dimensions_consistent(self, tokenizer, image_batch_samples):
        """Every output tensor must have the same leading batch dimension."""
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer,
            font_path=FONT_PATH,
            width=64,
            height=64,
            normalize_image=False,
        )
        result = processor(batch=image_batch_samples)
        batch_size = len(image_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )
