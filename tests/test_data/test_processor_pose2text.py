"""Tests for Pose2TextTranslationProcessor."""

import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.pose2text_preprocessor import (
    Pose2TextTranslationProcessor,
)


class TestPoseFileToTensor:
    def test_reads_pose_file(self, tokenizer, dummy_pose_file):
        processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
        )
        tensor = processor._pose_file_to_tensor(dummy_pose_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 2  # (frames, features)
        assert tensor.shape[0] > 0  # should have some frames

    def test_tensor_passthrough(self, tokenizer):
        """If input is already a tensor, it should be returned as-is."""
        processor = Pose2TextTranslationProcessor(tokenizer=tokenizer)
        t = torch.randn(10, 64)
        result = processor._pose_file_to_tensor(t)
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
        tensor_full = processor_no_skip._pose_file_to_tensor(dummy_pose_file)
        tensor_skip = processor_skip._pose_file_to_tensor(dummy_pose_file)
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
        tensor_reduced = processor_reduced._pose_file_to_tensor(dummy_pose_file)
        tensor_full = processor_full._pose_file_to_tensor(dummy_pose_file)
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
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
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
