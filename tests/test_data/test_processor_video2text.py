"""Tests for Video2TextTranslationProcessor."""

import numpy as np
import torch

from multimodalhugs.processors.video2text_preprocessor import (
    Video2TextTranslationProcessor,
)


class TestVideoFileToTensor:
    def test_reads_video_file(self, tokenizer, dummy_video_file):
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        tensor = processor._video_file_to_tensor(dummy_video_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4  # (T, C, H, W)
        assert tensor.shape[0] > 0  # should have some frames

    def test_tensor_passthrough(self, tokenizer):
        """If input is already a tensor, it should be returned as-is."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        t = torch.randn(10, 3, 64, 64)
        result = processor._video_file_to_tensor(t)
        assert torch.equal(result, t)

    def test_ndarray_to_tensor(self, tokenizer):
        """If input is an ndarray, it should be converted to tensor."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        arr = np.random.rand(10, 3, 64, 64).astype(np.float32)
        result = processor._video_file_to_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 3, 64, 64)

    def test_skip_frames_stride(self, tokenizer, dummy_video_file):
        """skip_frames_stride should reduce temporal dimension."""
        processor_no_skip = Video2TextTranslationProcessor(tokenizer=tokenizer)
        processor_skip = Video2TextTranslationProcessor(
            tokenizer=tokenizer, skip_frames_stride=2
        )
        tensor_full = processor_no_skip._video_file_to_tensor(dummy_video_file)
        tensor_skip = processor_skip._video_file_to_tensor(dummy_video_file)
        assert tensor_skip.shape[0] < tensor_full.shape[0]


class TestVideoObtainMultimodalInputAndMasks:
    def test_returns_input_frames_and_mask(self, tokenizer, dummy_video_file):
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        batch = [
            {
                "signal": dummy_video_file,
                "signal_start": 0,
                "signal_end": 0,
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "test",
            },
        ]
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
        assert "input_frames" in result
        assert "attention_mask" in result

    def test_join_chw_flattens(self, tokenizer, dummy_video_file):
        """join_chw=True should flatten C*H*W into last dimension."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer, join_chw=True)
        batch = [
            {
                "signal": dummy_video_file,
                "signal_start": 0,
                "signal_end": 0,
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "test",
            },
        ]
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
        # With join_chw, shape should be [B, T, C*H*W]
        assert result["input_frames"].ndim == 3


class TestVideoTransformGetItemsOutput:
    def test_converts_signals_to_tensors(self, tokenizer, dummy_video_file):
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        batch = {
            "signal": [dummy_video_file],
            "signal_start": [0],
            "signal_end": [0],
        }
        result = processor._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)
