"""Tests for Pose2TextTranslationProcessor."""

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.legacy.pose2text_preprocessor import (
    Pose2TextTranslationProcessor,
)
from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor


def _modality_proc(processor):
    """Return the underlying PoseModalityProcessor from the wrapper."""
    return processor.slots[0].processor


class TestPoseFileToTensor:
    def test_reads_pose_file(self, tokenizer, dummy_pose_file):
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
        )
        tensor = _modality_proc(processor).process_sample(dummy_pose_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 2  # (frames, features)
        assert tensor.shape[0] > 0  # should have some frames

    def test_tensor_passthrough(self, tokenizer):
        """If input is already a tensor, it should be returned as-is."""
        processor = Pose2TextTranslationProcessor(tokenizer=tokenizer)
        t = torch.randn(10, 64)
        result = _modality_proc(processor).process_sample(t)
        assert torch.equal(result, t)

    def test_skip_frames_stride(self, tokenizer, dummy_pose_file):
        """skip_frames_stride should reduce temporal dimension."""
        processor_no_skip = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
        )
        processor_skip = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
            skip_frames_stride=2,
        )
        tensor_full = _modality_proc(processor_no_skip).process_sample(dummy_pose_file)
        tensor_skip = _modality_proc(processor_skip).process_sample(dummy_pose_file)
        # Skipped version should have roughly half the frames
        assert tensor_skip.shape[0] <= (tensor_full.shape[0] + 1) // 2 + 1
        assert tensor_skip.shape[0] < tensor_full.shape[0]

    def test_reduce_holistic_false(self, tokenizer, dummy_pose_file):
        """With reduce_holistic_poses=False, more features per frame."""
        processor_reduced = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
        )
        processor_full = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=False,
        )
        tensor_reduced = _modality_proc(processor_reduced).process_sample(dummy_pose_file)
        tensor_full = _modality_proc(processor_full).process_sample(dummy_pose_file)
        # Full should have more features per frame than reduced
        assert tensor_full.shape[1] > tensor_reduced.shape[1]


class TestPoseObtainMultimodalInputAndMasks:
    def test_returns_input_frames_and_mask(self, tokenizer, dummy_pose_file):
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
        )
        batch = [
            {
                "signal": dummy_pose_file,
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
        assert result["input_frames"].ndim == 3  # (batch, frames, features)
        assert result["attention_mask"].ndim == 2  # (batch, frames)


class TestPoseTransformGetItemsOutput:
    def test_converts_signals_to_tensors(self, tokenizer, dummy_pose_file):
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
        )
        batch = {
            "signal": [dummy_pose_file],
            "signal_start": [0],
            "signal_end": [0],
        }
        result = processor._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)


class TestPoseProcessorCall:
    """Full __call__() tests — the path exercised by DataCollatorMultimodalSeq2Seq."""

    EXPECTED_KEYS = {
        "input_frames",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    }

    def test_returns_batch_feature(self, tokenizer, pose_batch_samples):
        """__call__ should return a BatchFeature (HF-compatible mapping)."""
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer, reduce_holistic_poses=True
        )
        result = processor(batch=pose_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_has_all_expected_keys(self, tokenizer, pose_batch_samples):
        """Output must contain all keys consumed by the model forward()."""
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer, reduce_holistic_poses=True
        )
        result = processor(batch=pose_batch_samples)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_batch_dimensions_consistent(self, tokenizer, pose_batch_samples):
        """Every output tensor must have the same leading batch dimension."""
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer, reduce_holistic_poses=True
        )
        result = processor(batch=pose_batch_samples)
        batch_size = len(pose_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )


class TestPoseSignalStartEndUnit:
    """Tests for the signal_start_end_unit parameter on PoseModalityProcessor."""

    def test_default_unit_is_milliseconds(self, dummy_pose_file):
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        assert proc.signal_start_end_unit == "milliseconds"

    def test_frames_unit_accepted(self, dummy_pose_file):
        proc = PoseModalityProcessor(
            reduce_holistic_poses=True,
            signal_start_end_unit="frames",
        )
        tensor = proc.process_sample(dummy_pose_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 2

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="signal_start_end_unit"):
            PoseModalityProcessor(signal_start_end_unit="seconds")

    def test_frames_unit_slices_output(self, dummy_pose_file):
        """Requesting frames 0..5 should yield fewer frames than the full file."""
        proc_full = PoseModalityProcessor(
            reduce_holistic_poses=True,
            signal_start_end_unit="frames",
        )
        proc_sliced = PoseModalityProcessor(
            reduce_holistic_poses=True,
            signal_start_end_unit="frames",
        )
        # dummy_pose_file has 10 frames; request first 5
        tensor_full = proc_full.process_sample(dummy_pose_file)
        tensor_sliced = proc_sliced.process_sample(
            {"signal": dummy_pose_file, "signal_start": 0, "signal_end": 5}
        )
        assert tensor_sliced.shape[0] == 5
        assert tensor_full.shape[0] == 10

    def test_frames_unit_zero_zero_loads_full_file(self, dummy_pose_file):
        """signal_start=0, signal_end=0 with unit='frames' loads the full file."""
        proc = PoseModalityProcessor(
            reduce_holistic_poses=True,
            signal_start_end_unit="frames",
        )
        tensor = proc.process_sample(
            {"signal": dummy_pose_file, "signal_start": 0, "signal_end": 0}
        )
        assert tensor.shape[0] == 10  # all frames present

    def test_milliseconds_and_frames_unit_same_full_load(self, dummy_pose_file):
        """Both units with start=0, end=0 should yield the same frame count."""
        proc_ms = PoseModalityProcessor(
            reduce_holistic_poses=True,
            signal_start_end_unit="milliseconds",
        )
        proc_fr = PoseModalityProcessor(
            reduce_holistic_poses=True,
            signal_start_end_unit="frames",
        )
        t_ms = proc_ms.process_sample(
            {"signal": dummy_pose_file, "signal_start": 0, "signal_end": 0}
        )
        t_fr = proc_fr.process_sample(
            {"signal": dummy_pose_file, "signal_start": 0, "signal_end": 0}
        )
        assert t_ms.shape[0] == t_fr.shape[0]

    def test_legacy_wrapper_passes_unit_through(self, tokenizer, dummy_pose_file):
        """Pose2TextTranslationProcessor should propagate signal_start_end_unit."""
        proc = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            signal_start_end_unit="frames",
        )
        assert _modality_proc(proc).signal_start_end_unit == "frames"
