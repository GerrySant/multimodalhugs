"""
Regression tests for processor outputs.

Each test runs a processor with the real committed asset files and compares the
output against the golden values captured by tests/assets/generate_golden.py.

If a test fails after an intentional preprocessing change, regenerate the
golden files:
    python tests/assets/generate_golden.py

If a test fails unexpectedly, that signals an unintended change in behaviour.

Golden values capture per tensor:
  - shape   (must match exactly)
  - dtype   (must match exactly)
  - For float tensors: mean, std, min, max, sum  (within a tight tolerance)
  - For integer tensors: sum and full values      (must match exactly)
"""

import json
import os

import pytest
import torch

from tests.test_data.conftest import ASSETS_DIR, CLIP_PROCESSOR_PATH, FONT_PATH, TINY_TOKENIZER_PATH
from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from multimodalhugs.processors.features_modality_processor import FeaturesModalityProcessor
from multimodalhugs.processors.image_modality_processor import ImageModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor

GOLDEN_DIR = os.path.join(ASSETS_DIR, "golden")

FLOAT_ABS_TOL = 1e-4   # tolerance for floating-point statistics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden(name: str) -> dict:
    """Load a golden JSON file by modality name; skip the test if absent."""
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Golden file not found: {path}  Run tests/assets/generate_golden.py first.")
    with open(path) as f:
        return json.load(f)


def assert_matches_golden(tensor: torch.Tensor, golden: dict, key: str,
                          abs_tol: float = FLOAT_ABS_TOL):
    """
    Compare a tensor against its golden summary.

    Checks exact shape and dtype.  For float tensors also checks mean, std,
    min, max, sum (within abs_tol) and a per-element prefix/suffix fingerprint
    to catch permutations.  For integer tensors checks exact sum and full
    values list.  abs_tol defaults to FLOAT_ABS_TOL but can be overridden per
    golden file via the top-level "abs_tol" key (used for platform-sensitive
    processors such as video where PyAV decoding varies slightly across OSes).
    """
    assert list(tensor.shape) == golden["shape"], \
        f"[{key}] shape mismatch: got {list(tensor.shape)}, expected {golden['shape']}"
    assert str(tensor.dtype) == golden["dtype"], \
        f"[{key}] dtype mismatch: got {tensor.dtype}, expected {golden['dtype']}"

    if tensor.is_floating_point():
        f = tensor.float()
        flat = f.flatten()
        assert f.mean().item() == pytest.approx(golden["mean"], abs=abs_tol), \
            f"[{key}] mean mismatch"
        assert f.std().item()  == pytest.approx(golden["std"],  abs=abs_tol), \
            f"[{key}] std mismatch"
        assert f.min().item()  == pytest.approx(golden["min"],  abs=abs_tol), \
            f"[{key}] min mismatch"
        assert f.max().item()  == pytest.approx(golden["max"],  abs=abs_tol), \
            f"[{key}] max mismatch"
        # Scale sum tolerance by number of elements so per-element precision
        # stays consistent with mean/std/min/max checks.
        assert f.sum().item()  == pytest.approx(golden["sum"],  abs=abs_tol * tensor.numel()), \
            f"[{key}] sum mismatch"
        if "first_values" in golden:
            assert flat[:8].tolist() == pytest.approx(golden["first_values"], abs=abs_tol), \
                f"[{key}] first_values mismatch"
        if "last_values" in golden:
            assert flat[-8:].tolist() == pytest.approx(golden["last_values"], abs=abs_tol), \
                f"[{key}] last_values mismatch"
    else:
        assert int(tensor.sum().item()) == golden["sum"], \
            f"[{key}] sum mismatch"
        assert tensor.tolist() == golden["values"], \
            f"[{key}] values mismatch"


def check_all_keys(result, golden):
    """Assert every key in the golden file is present and matches in result."""
    abs_tol = golden.get("abs_tol", FLOAT_ABS_TOL)
    for key, gval in golden.items():
        if key == "abs_tol":
            continue
        assert key in result, f"Missing key '{key}' in processor output"
        assert_matches_golden(result[key], gval, key, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# pose2text
# ---------------------------------------------------------------------------

class TestPose2TextRegression:
    """Regression tests for Pose2TextTranslationProcessor using committed pose assets."""

    def test_output_matches_golden(self, tokenizer, pose_asset_samples):
        from multimodalhugs.processors.legacy.pose2text_preprocessor import Pose2TextTranslationProcessor
        processor = Pose2TextTranslationProcessor(tokenizer=tokenizer, reduce_holistic_poses=True)
        result = processor(batch=pose_asset_samples)
        check_all_keys(result, load_golden("pose2text"))


# ---------------------------------------------------------------------------
# video2text
# ---------------------------------------------------------------------------

class TestVideo2TextRegression:
    """Regression tests for Video2TextTranslationProcessor using committed video assets."""

    def test_output_matches_golden(self, tokenizer, video_asset_samples):
        from multimodalhugs.processors.legacy.video2text_preprocessor import Video2TextTranslationProcessor
        processor = Video2TextTranslationProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path=CLIP_PROCESSOR_PATH,
        )
        result = processor(batch=video_asset_samples)
        check_all_keys(result, load_golden("video2text"))


# ---------------------------------------------------------------------------
# features2text
# ---------------------------------------------------------------------------

class TestFeatures2TextRegression:
    """Regression tests for Features2TextTranslationProcessor using committed .npy assets."""

    def test_output_matches_golden(self, tokenizer, features_asset_samples):
        from multimodalhugs.processors.legacy.features2text_preprocessor import Features2TextTranslationProcessor
        processor = Features2TextTranslationProcessor(tokenizer=tokenizer, use_cache=False)
        result = processor(batch=features_asset_samples)
        check_all_keys(result, load_golden("features2text"))


# ---------------------------------------------------------------------------
# text2text
# ---------------------------------------------------------------------------

class TestText2TextRegression:
    """Regression tests for Text2TextTranslationProcessor using inline text samples."""

    def test_output_matches_golden(self, tokenizer, text_asset_samples):
        from multimodalhugs.processors.legacy.text2text_preprocessor import Text2TextTranslationProcessor
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_asset_samples)
        check_all_keys(result, load_golden("text2text"))


# ---------------------------------------------------------------------------
# labels  (create_seq2seq_labels_from_samples — will move to TextModalityProcessor)
# ---------------------------------------------------------------------------

class TestLabelsRegression:
    """
    Regression tests for create_seq2seq_labels_from_samples.

    This function currently lives in DataCollatorMultimodalSeq2Seq but will move
    into TextModalityProcessor(role="target") during the processor refactoring.
    The golden file defines the exact contract that the new implementation must satisfy.
    """

    def test_output_matches_golden(self, tokenizer, text_asset_samples):
        from multimodalhugs.data.datacollators.multimodal_datacollator import (
            create_seq2seq_labels_from_samples,
        )
        result = create_seq2seq_labels_from_samples(
            samples=text_asset_samples,
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            padding=True,
            return_tensors="pt",
        )
        check_all_keys(result, load_golden("labels"))


# ---------------------------------------------------------------------------
# image2text
# ---------------------------------------------------------------------------

class TestImage2TextRegression:
    """Regression tests for Image2TextTranslationProcessor using inline text rendered to images."""

    def test_output_matches_golden(self, tokenizer, image_asset_samples):
        from multimodalhugs.processors.legacy.image2text_preprocessor import Image2TextTranslationProcessor
        processor = Image2TextTranslationProcessor(
            tokenizer=tokenizer, font_path=FONT_PATH,
            width=224, height=224, normalize_image=False,
        )
        result = processor(batch=image_asset_samples)
        check_all_keys(result, load_golden("image2text"))


# ---------------------------------------------------------------------------
# Flat-slots MultimodalMetaProcessor — golden parity with legacy wrappers
# ---------------------------------------------------------------------------

def _text_slots(tokenizer):
    """The three text slots shared by all modalities (label, encoder prompt, decoder prompt)."""
    return [
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
            output_data_key="encoder_prompt",
            output_mask_key="encoder_prompt_length_padding_mask",
            column_map={"encoder_prompt": "signal"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
            output_data_key="decoder_input_ids",
            output_mask_key="decoder_attention_mask",
            column_map={"decoder_prompt": "signal"},
        ),
    ]


class TestMetaProcessorPose2TextGolden:
    """MultimodalMetaProcessor(slots=[PoseModalityProcessor, ...]) must match the pose2text golden."""

    def test_output_matches_golden(self, tokenizer, pose_asset_samples):
        processor = MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=PoseModalityProcessor(reduce_holistic_poses=True),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                    column_map={
                        "signal": "signal",
                        "signal_start": "signal_start",
                        "signal_end": "signal_end",
                    },
                ),
                *_text_slots(tokenizer),
            ],
            tokenizer=tokenizer,
        )
        result = processor(batch=pose_asset_samples)
        check_all_keys(result, load_golden("pose2text"))


class TestMetaProcessorVideo2TextGolden:
    """MultimodalMetaProcessor(slots=[VideoModalityProcessor, ...]) must match the video2text golden."""

    def test_output_matches_golden(self, tokenizer, video_asset_samples):
        processor = MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=VideoModalityProcessor(
                        custom_preprocessor_path=CLIP_PROCESSOR_PATH,
                        use_cache=True,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                    column_map={
                        "signal": "signal",
                        "signal_start": "signal_start",
                        "signal_end": "signal_end",
                    },
                ),
                *_text_slots(tokenizer),
            ],
            tokenizer=tokenizer,
        )
        result = processor(batch=video_asset_samples)
        check_all_keys(result, load_golden("video2text"))


class TestMetaProcessorFeatures2TextGolden:
    """MultimodalMetaProcessor(slots=[FeaturesModalityProcessor, ...]) must match the features2text golden."""

    def test_output_matches_golden(self, tokenizer, features_asset_samples):
        processor = MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=FeaturesModalityProcessor(use_cache=False),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                ),
                *_text_slots(tokenizer),
            ],
            tokenizer=tokenizer,
        )
        result = processor(batch=features_asset_samples)
        check_all_keys(result, load_golden("features2text"))


class TestMetaProcessorText2TextGolden:
    """MultimodalMetaProcessor(slots=[TextModalityProcessor(encoder), ...]) must match the text2text golden."""

    def test_output_matches_golden(self, tokenizer, text_asset_samples):
        processor = MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                    output_data_key="input_ids",
                    output_mask_key="attention_mask",
                    column_map={"signal": "signal"},
                ),
                *_text_slots(tokenizer),
            ],
            tokenizer=tokenizer,
        )
        result = processor(batch=text_asset_samples)
        check_all_keys(result, load_golden("text2text"))


class TestMetaProcessorImage2TextGolden:
    """MultimodalMetaProcessor(slots=[ImageModalityProcessor, ...]) must match the image2text golden."""

    def test_output_matches_golden(self, tokenizer, image_asset_samples):
        processor = MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=ImageModalityProcessor(
                        font_path=FONT_PATH,
                        width=224,
                        height=224,
                        normalize_image=False,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                ),
                *_text_slots(tokenizer),
            ],
            tokenizer=tokenizer,
        )
        result = processor(batch=image_asset_samples)
        check_all_keys(result, load_golden("image2text"))
