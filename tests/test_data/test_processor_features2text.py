"""Tests for Features2TextTranslationProcessor."""

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.legacy.features2text_preprocessor import (
    Features2TextTranslationProcessor,
)


def _modality_proc(processor):
    """Return the underlying FeaturesModalityProcessor from the wrapper."""
    return processor.slots[0].processor


class TestFeaturesFileToTensor:
    def test_from_npy_path(self, tokenizer, dummy_npy_file):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        tensor = _modality_proc(processor).process_sample(dummy_npy_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 64)

    def test_from_ndarray(self, tokenizer):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        arr = np.random.rand(8, 32).astype(np.float32)
        tensor = _modality_proc(processor).process_sample(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (8, 32)

    def test_from_tensor(self, tokenizer):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        t = torch.randn(5, 16)
        result = _modality_proc(processor).process_sample(t)
        assert torch.equal(result, t)

    def test_from_nested_list(self, tokenizer):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        tensor = _modality_proc(processor).process_sample(data)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2)

    def test_skip_frames_stride(self, tokenizer, dummy_npy_file):
        """skip_frames_stride should reduce temporal dimension."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False, skip_frames_stride=3
        )
        tensor = _modality_proc(processor).process_sample(dummy_npy_file)
        # Original has 10 frames, stride 3 → ceil(10/3) = 4 frames
        assert tensor.shape[0] == 4

    def test_temporal_dimension_position(self, tokenizer, tmp_path):
        """temporal_dimention_position moves correct axis to position 0."""
        path = str(tmp_path / "transposed.npy")
        # Shape (64, 10) where temporal dim is at position 1
        np.save(path, np.random.rand(64, 10).astype(np.float32))
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False, temporal_dimention_position=1
        )
        tensor = _modality_proc(processor).process_sample(path)
        # After movedim(1, 0), shape should be (10, 64)
        assert tensor.shape == (10, 64)


class TestFeaturesObtainMultimodalInputAndMasks:
    def test_returns_input_frames_and_mask(self, tokenizer, dummy_npy_file):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        batch = [
            {
                "signal": dummy_npy_file,
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "test",
            },
        ]
        result = processor(batch=batch)
        assert "input_frames" in result
        assert "attention_mask" in result

    def test_padding_different_lengths(self, tokenizer, tmp_path):
        """Test padding when features have different temporal lengths."""
        path1 = str(tmp_path / "feat1.npy")
        path2 = str(tmp_path / "feat2.npy")
        np.save(path1, np.random.rand(5, 16).astype(np.float32))
        np.save(path2, np.random.rand(10, 16).astype(np.float32))
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        batch = [
            {
                "signal": path1,
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "a",
            },
            {
                "signal": path2,
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "b",
            },
        ]
        result = processor(batch=batch)
        assert result["input_frames"].shape == (2, 10, 16)
        assert result["attention_mask"].shape == (2, 10)
        # First sample: 5 real, 5 padded
        assert result["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert result["attention_mask"][1].tolist() == [1] * 10


class TestFeaturesTransformGetItemsOutput:
    def test_converts_signals_to_tensors(self, tokenizer, dummy_npy_file):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        batch = {
            "signal": [dummy_npy_file, dummy_npy_file],
        }
        result = processor._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)
        assert isinstance(result["signal"][1], torch.Tensor)


class TestFeaturesCacheBehavior:
    def test_cache_enabled(self, tokenizer, dummy_npy_file):
        """With use_cache=True, the underlying processor should initialize cache."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=True
        )
        assert hasattr(_modality_proc(processor), "_cache_size")
        # Should still work to load files
        tensor = _modality_proc(processor).process_sample(dummy_npy_file)
        assert isinstance(tensor, torch.Tensor)


class TestFeaturesProcessorCall:
    """Full __call__() tests — the path exercised by DataCollatorMultimodalSeq2Seq."""

    EXPECTED_KEYS = {
        "input_frames",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    }

    def test_returns_batch_feature(self, tokenizer, features_batch_samples):
        """__call__ should return a BatchFeature (HF-compatible mapping)."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        result = processor(batch=features_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_has_all_expected_keys(self, tokenizer, features_batch_samples):
        """Output must contain all keys consumed by the model forward()."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        result = processor(batch=features_batch_samples)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_batch_dimensions_consistent(self, tokenizer, features_batch_samples):
        """Every output tensor must have the same leading batch dimension."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        result = processor(batch=features_batch_samples)
        batch_size = len(features_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )
