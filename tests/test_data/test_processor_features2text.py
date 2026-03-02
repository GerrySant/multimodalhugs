"""Tests for Features2TextTranslationProcessor."""

import numpy as np
import torch

from multimodalhugs.processors.features2text_preprocessor import (
    Features2TextTranslationProcessor,
)


class TestFeaturesFileToTensor:
    def test_from_npy_path(self, tokenizer, dummy_npy_file):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        tensor = processor._features_file_to_tensor(dummy_npy_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 64)

    def test_from_ndarray(self, tokenizer):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        arr = np.random.rand(8, 32).astype(np.float32)
        tensor = processor._features_file_to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (8, 32)

    def test_from_tensor(self, tokenizer):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        t = torch.randn(5, 16)
        result = processor._features_file_to_tensor(t)
        assert torch.equal(result, t)

    def test_from_nested_list(self, tokenizer):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        tensor = processor._features_file_to_tensor(data)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2)

    def test_skip_frames_stride(self, tokenizer, dummy_npy_file):
        """skip_frames_stride should reduce temporal dimension."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False, skip_frames_stride=3
        )
        tensor = processor._features_file_to_tensor(dummy_npy_file)
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
        tensor = processor._features_file_to_tensor(path)
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
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
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
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
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
        """With use_cache=True, processor should initialize cache."""
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=True
        )
        assert hasattr(processor, "_cache_size")
        # Should still work to load files
        tensor = processor._features_file_to_tensor(dummy_npy_file)
        assert isinstance(tensor, torch.Tensor)
