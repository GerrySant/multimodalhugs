"""Tests for Video2TextTranslationProcessor."""

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.legacy.video2text_preprocessor import (
    Video2TextTranslationProcessor,
)
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from tests.test_data.conftest import CLIP_PROCESSOR_PATH


def _modality_proc(processor):
    """Return the underlying VideoModalityProcessor from the wrapper."""
    return processor.slots[0].processor


class TestVideoFileToTensor:
    def test_reads_video_file(self, tokenizer, dummy_video_file):
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        tensor = _modality_proc(processor).process_sample(dummy_video_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4  # (T, C, H, W)
        assert tensor.shape[0] > 0  # should have some frames

    def test_tensor_passthrough(self, tokenizer):
        """If input is already a tensor, it should be returned as-is."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        t = torch.randn(10, 3, 64, 64)
        result = _modality_proc(processor).process_sample(t)
        assert torch.equal(result, t)

    def test_ndarray_to_tensor(self, tokenizer):
        """If input is an ndarray, it should be converted to tensor."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        arr = np.random.rand(10, 3, 64, 64).astype(np.float32)
        result = _modality_proc(processor).process_sample(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 3, 64, 64)

    def test_skip_frames_stride(self, tokenizer, dummy_video_file):
        """skip_frames_stride should reduce temporal dimension."""
        processor_no_skip = Video2TextTranslationProcessor(tokenizer=tokenizer)
        processor_skip = Video2TextTranslationProcessor(
            tokenizer=tokenizer, skip_frames_stride=2
        )
        tensor_full = _modality_proc(processor_no_skip).process_sample(dummy_video_file)
        tensor_skip = _modality_proc(processor_skip).process_sample(dummy_video_file)
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
        result = processor(batch=batch)
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
        result = processor(batch=batch)
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


class TestVideoProcessorCall:
    """Full __call__() tests — the path exercised by DataCollatorMultimodalSeq2Seq."""

    EXPECTED_KEYS = {
        "input_frames",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    }

    def test_returns_batch_feature(self, tokenizer, video_batch_samples):
        """__call__ should return a BatchFeature (HF-compatible mapping)."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=video_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_has_all_expected_keys(self, tokenizer, video_batch_samples):
        """Output must contain all keys consumed by the model forward()."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=video_batch_samples)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_batch_dimensions_consistent(self, tokenizer, video_batch_samples):
        """Every output tensor must have the same leading batch dimension."""
        processor = Video2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=video_batch_samples)
        batch_size = len(video_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )


class TestVideoSignalStartEndUnit:
    """Tests for the signal_start_end_unit parameter on VideoModalityProcessor."""

    def test_default_unit_is_milliseconds(self):
        proc = VideoModalityProcessor()
        assert proc.signal_start_end_unit == "milliseconds"

    def test_frames_unit_accepted(self, dummy_video_file):
        proc = VideoModalityProcessor(signal_start_end_unit="frames")
        tensor = proc.process_sample(dummy_video_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4  # (T, C, H, W)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="signal_start_end_unit"):
            VideoModalityProcessor(signal_start_end_unit="seconds")

    def test_frames_unit_slices_output(self, dummy_video_file):
        """Requesting frames 0..5 should yield fewer frames than the full file."""
        proc = VideoModalityProcessor(signal_start_end_unit="frames")
        tensor_full = proc.process_sample(dummy_video_file)
        tensor_sliced = proc.process_sample(
            {"signal": dummy_video_file, "signal_start": 0, "signal_end": 5}
        )
        assert tensor_sliced.shape[0] == 5
        assert tensor_full.shape[0] > tensor_sliced.shape[0]

    def test_frames_unit_zero_zero_loads_full_file(self, dummy_video_file):
        """signal_start=0, signal_end=0 with unit='frames' loads the full file."""
        proc_ms = VideoModalityProcessor(signal_start_end_unit="milliseconds")
        proc_fr = VideoModalityProcessor(signal_start_end_unit="frames")
        t_ms = proc_ms.process_sample(
            {"signal": dummy_video_file, "signal_start": 0, "signal_end": 0}
        )
        t_fr = proc_fr.process_sample(
            {"signal": dummy_video_file, "signal_start": 0, "signal_end": 0}
        )
        assert t_ms.shape[0] == t_fr.shape[0]

    def test_legacy_wrapper_passes_unit_through(self, tokenizer, dummy_video_file):
        """Video2TextTranslationProcessor should propagate signal_start_end_unit."""
        proc = Video2TextTranslationProcessor(
            tokenizer=tokenizer,
            signal_start_end_unit="frames",
        )
        assert _modality_proc(proc).signal_start_end_unit == "frames"

    def test_opencv_path_frames_unit_slices_output(self, dummy_video_file):
        """OpenCV path: signal_start_end_unit='frames' should slice by frame index."""
        proc_full = VideoModalityProcessor(
            custom_preprocessor_path=CLIP_PROCESSOR_PATH,
            signal_start_end_unit="frames",
        )
        proc_sliced = VideoModalityProcessor(
            custom_preprocessor_path=CLIP_PROCESSOR_PATH,
            signal_start_end_unit="frames",
        )
        tensor_full = proc_full.process_sample(dummy_video_file)
        tensor_sliced = proc_sliced.process_sample(
            {"signal": dummy_video_file, "signal_start": 2, "signal_end": 7}
        )
        # dummy_video_file has 10 frames; requesting frames 2..7 should yield fewer
        assert tensor_sliced.shape[0] < tensor_full.shape[0]
        assert tensor_sliced.shape[0] <= 5  # at most 5 frames in [2, 7)
