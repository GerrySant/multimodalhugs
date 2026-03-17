"""
Dataset-to-processor contract tests.

For each modality, these tests verify that the dict yielded by
``_generate_examples()`` contains *at least* the keys and value types that the
processor regression-test fixtures assume.

If this contract holds, then swapping a hand-crafted fixture dict for a real
Dataset sample will not break the processor.

TSV files for path-dependent modalities (pose, video, features) are generated
by ``tests/assets/generate_assets.py`` and are not committed to git (they
contain absolute paths).  Tests for those modalities are skipped automatically
when the TSV has not been generated yet.
"""

import os

import pytest

from tests.test_data.conftest import ASSETS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POSE_TSV     = os.path.join(ASSETS_DIR, "pose",     "metadata.tsv")
VIDEO_TSV    = os.path.join(ASSETS_DIR, "video",    "metadata.tsv")
FEATURES_TSV = os.path.join(ASSETS_DIR, "features", "metadata.tsv")
TEXT_TSV     = os.path.join(ASSETS_DIR, "text",     "metadata.tsv")
IMAGE_TSV    = os.path.join(ASSETS_DIR, "image",    "metadata.tsv")


def _require_tsv(path):
    """Skip the test if the TSV has not been generated yet."""
    if not os.path.exists(path):
        pytest.skip(
            f"Asset TSV not found: {path}  "
            "Run tests/assets/generate_assets.py first."
        )


def _first_example(dataset_instance, tsv_path, split="train"):
    """Return the first (key, sample) pair from _generate_examples()."""
    examples = list(
        dataset_instance._generate_examples(
            split=split, metafile_path=tsv_path
        )
    )
    assert len(examples) > 0, "Dataset yielded no examples"
    return examples[0][1]  # (idx, sample) → sample


def _assert_key_type(sample, key, expected_type, modality):
    assert key in sample, (
        f"[{modality}] key '{key}' missing from dataset sample. "
        f"Available keys: {list(sample.keys())}"
    )
    assert isinstance(sample[key], expected_type), (
        f"[{modality}] key '{key}' has type {type(sample[key]).__name__}, "
        f"expected {expected_type.__name__}"
    )


# ---------------------------------------------------------------------------
# pose2text
# ---------------------------------------------------------------------------

class TestPose2TextDatasetProcessorContract:
    """
    The pose regression fixture assumes:
        signal        str  (path to .pose file)
        signal_start  int
        signal_end    int
        encoder_prompt str
        decoder_prompt str
        output        str
    """

    def test_dataset_yields_required_keys_and_types(self):
        _require_tsv(POSE_TSV)
        from multimodalhugs.data.datasets.pose2text import (
            Pose2TextDataset, Pose2TextDataConfig,
        )
        ds = Pose2TextDataset(config=Pose2TextDataConfig())
        sample = _first_example(ds, POSE_TSV)

        for key, typ in [
            ("signal",         str),
            ("signal_start",   int),
            ("signal_end",     int),
            ("encoder_prompt", str),
            ("decoder_prompt", str),
            ("output",         str),
        ]:
            _assert_key_type(sample, key, typ, "pose2text")

    def test_signal_is_existing_file(self):
        _require_tsv(POSE_TSV)
        from multimodalhugs.data.datasets.pose2text import (
            Pose2TextDataset, Pose2TextDataConfig,
        )
        ds = Pose2TextDataset(config=Pose2TextDataConfig())
        sample = _first_example(ds, POSE_TSV)
        assert os.path.exists(sample["signal"]), (
            f"signal path does not exist: {sample['signal']}"
        )


# ---------------------------------------------------------------------------
# video2text
# ---------------------------------------------------------------------------

class TestVideo2TextDatasetProcessorContract:
    """
    The video regression fixture assumes:
        signal        str  (path to .mp4 file)
        signal_start  int
        signal_end    int
        encoder_prompt str
        decoder_prompt str
        output        str
    """

    def test_dataset_yields_required_keys_and_types(self):
        _require_tsv(VIDEO_TSV)
        from multimodalhugs.data.datasets.video2text import (
            Video2TextDataset, Video2TextDataConfig,
        )
        ds = Video2TextDataset(config=Video2TextDataConfig())
        sample = _first_example(ds, VIDEO_TSV)

        for key, typ in [
            ("signal",         str),
            ("signal_start",   int),
            ("signal_end",     int),
            ("encoder_prompt", str),
            ("decoder_prompt", str),
            ("output",         str),
        ]:
            _assert_key_type(sample, key, typ, "video2text")

    def test_signal_is_existing_file(self):
        _require_tsv(VIDEO_TSV)
        from multimodalhugs.data.datasets.video2text import (
            Video2TextDataset, Video2TextDataConfig,
        )
        ds = Video2TextDataset(config=Video2TextDataConfig())
        sample = _first_example(ds, VIDEO_TSV)
        assert os.path.exists(sample["signal"]), (
            f"signal path does not exist: {sample['signal']}"
        )


# ---------------------------------------------------------------------------
# features2text
# ---------------------------------------------------------------------------

class TestFeatures2TextDatasetProcessorContract:
    """
    The features regression fixture assumes:
        signal        str  (path to .npy file)
        signal_start  int
        signal_end    int
        encoder_prompt str
        decoder_prompt str
        output        str
    """

    def test_dataset_yields_required_keys_and_types(self):
        _require_tsv(FEATURES_TSV)
        from multimodalhugs.data.datasets.features2text import (
            Features2TextDataset, Features2TextDataConfig,
        )
        ds = Features2TextDataset(config=Features2TextDataConfig())
        sample = _first_example(ds, FEATURES_TSV)

        for key, typ in [
            ("signal",         str),
            ("signal_start",   int),
            ("signal_end",     int),
            ("encoder_prompt", str),
            ("decoder_prompt", str),
            ("output",         str),
        ]:
            _assert_key_type(sample, key, typ, "features2text")

    def test_signal_is_existing_file(self):
        _require_tsv(FEATURES_TSV)
        from multimodalhugs.data.datasets.features2text import (
            Features2TextDataset, Features2TextDataConfig,
        )
        ds = Features2TextDataset(config=Features2TextDataConfig())
        sample = _first_example(ds, FEATURES_TSV)
        assert os.path.exists(sample["signal"]), (
            f"signal path does not exist: {sample['signal']}"
        )


# ---------------------------------------------------------------------------
# text2text
# ---------------------------------------------------------------------------

class TestText2TextDatasetProcessorContract:
    """
    The text regression fixture assumes:
        signal        str  (raw text)
        encoder_prompt str
        decoder_prompt str
        output        str
    """

    def test_dataset_yields_required_keys_and_types(self):
        from multimodalhugs.data.datasets.bilingual_text2text import (
            BilingualText2TextDataset, BilingualText2textMTDataConfig,
        )
        ds = BilingualText2TextDataset(config=BilingualText2textMTDataConfig())
        sample = _first_example(ds, TEXT_TSV)

        for key, typ in [
            ("signal",         str),
            ("encoder_prompt", str),
            ("decoder_prompt", str),
            ("output",         str),
        ]:
            _assert_key_type(sample, key, typ, "text2text")

    def test_signal_is_non_empty_string(self):
        from multimodalhugs.data.datasets.bilingual_text2text import (
            BilingualText2TextDataset, BilingualText2textMTDataConfig,
        )
        ds = BilingualText2TextDataset(config=BilingualText2textMTDataConfig())
        sample = _first_example(ds, TEXT_TSV)
        assert len(sample["signal"]) > 0, "signal string is empty"


# ---------------------------------------------------------------------------
# image2text
# ---------------------------------------------------------------------------

class TestImage2TextDatasetProcessorContract:
    """
    The image regression fixture assumes:
        signal        str  (raw text rendered to image by the processor)
        encoder_prompt str
        decoder_prompt str
        output        str

    The dataset may additionally yield signal_start / signal_end (set to 0),
    which the processor ignores.  Those extra keys do not break the contract.
    """

    def test_dataset_yields_required_keys_and_types(self):
        from multimodalhugs.data.datasets.bilingual_image2text import (
            BilingualImage2TextDataset, BilingualImage2textMTDataConfig,
        )
        ds = BilingualImage2TextDataset(
            config=BilingualImage2textMTDataConfig()
        )
        sample = _first_example(ds, IMAGE_TSV)

        for key, typ in [
            ("signal",         str),
            ("encoder_prompt", str),
            ("decoder_prompt", str),
            ("output",         str),
        ]:
            _assert_key_type(sample, key, typ, "image2text")

    def test_signal_matches_tsv_content(self):
        """The signal value should be the raw text string from the TSV."""
        from multimodalhugs.data.datasets.bilingual_image2text import (
            BilingualImage2TextDataset, BilingualImage2textMTDataConfig,
        )
        ds = BilingualImage2TextDataset(
            config=BilingualImage2textMTDataConfig()
        )
        sample = _first_example(ds, IMAGE_TSV)
        # The first row of image/metadata.tsv has this signal
        assert sample["signal"] == "Let's open Access."
