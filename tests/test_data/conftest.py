"""Shared fixtures for data loading tests."""

import os
import random

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Paths and sample data — single source of truth in tests/assets/sample_data.py
# ---------------------------------------------------------------------------
from tests.assets.sample_data import (
    ASSETS_DIR,
    TINY_TOKENIZER_PATH,
    FONT_PATH,
    CLIP_PROCESSOR_PATH,
    SIGNWRITING_STRINGS,
)


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_everything():
    """Set deterministic seeds for every test."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tokenizer():
    """Load the tiny T5-style tokenizer shipped with the test suite."""
    tok = PreTrainedTokenizerFast.from_pretrained(TINY_TOKENIZER_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Dummy media files  (written to tmp_path per test)
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_pose_file(tmp_path):
    """Create a minimal .pose file with 10 frames and standard holistic components."""
    from pose_format.utils.generic import fake_pose, get_standard_components_for_known_format

    components = get_standard_components_for_known_format("holistic")
    pose = fake_pose(num_frames=10, fps=25.0, components=components)
    path = tmp_path / "dummy.pose"
    with open(path, "wb") as f:
        pose.write(f)
    return str(path)


@pytest.fixture
def dummy_video_file(tmp_path):
    """Create a minimal 10-frame 64x64 MPEG4 video."""
    if not _AV_AVAILABLE:
        pytest.skip("av not installed")
    path = str(tmp_path / "dummy.mp4")
    container = av.open(path, mode="w")
    stream = container.add_stream("mpeg4", rate=25)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for i in range(10):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def dummy_npy_file(tmp_path):
    """Create a .npy file with shape (10, 64)."""
    path = str(tmp_path / "dummy.npy")
    np.save(path, np.random.rand(10, 64).astype(np.float32))
    return path


@pytest.fixture
def dummy_image_file(tmp_path):
    """Create a 64x64 RGB PNG image."""
    path = str(tmp_path / "dummy.png")
    Image.new("RGB", (64, 64), color="red").save(path)
    return path


# ---------------------------------------------------------------------------
# TSV fixtures  (one per modality, referencing dummy files)
# ---------------------------------------------------------------------------
def _write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")
    return str(path)


@pytest.fixture
def text2text_tsv(tmp_path):
    """TSV for BilingualText2TextDataset."""
    rows = [
        ("Hello world", "translate:", "de:", "Hallo Welt"),
        ("Good morning", "translate:", "de:", "Guten Morgen"),
        ("Thank you", "translate:", "de:", "Danke"),
    ]
    return _write_tsv(
        tmp_path / "text2text.tsv",
        "signal\tencoder_prompt\tdecoder_prompt\toutput",
        rows,
    )


@pytest.fixture
def pose2text_tsv(tmp_path, dummy_pose_file):
    """TSV for Pose2TextDataset pointing to the dummy pose file."""
    rows = [
        (dummy_pose_file, 0, 0, "translate:", "de:", "Hello"),
        (dummy_pose_file, 0, 0, "translate:", "de:", "World"),
        (dummy_pose_file, 0, 0, "translate:", "de:", "Test"),
    ]
    return _write_tsv(
        tmp_path / "pose2text.tsv",
        "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput",
        rows,
    )


@pytest.fixture
def video2text_tsv(tmp_path, dummy_video_file):
    """TSV for Video2TextDataset pointing to the dummy video file."""
    rows = [
        (dummy_video_file, 0, 0, "translate:", "de:", "Hello"),
        (dummy_video_file, 0, 0, "translate:", "de:", "World"),
    ]
    return _write_tsv(
        tmp_path / "video2text.tsv",
        "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput",
        rows,
    )


@pytest.fixture
def features2text_tsv(tmp_path, dummy_npy_file):
    """TSV for Features2TextDataset pointing to the dummy npy file."""
    rows = [
        (dummy_npy_file, 0, 0, "translate:", "de:", "Hello"),
        (dummy_npy_file, 0, 0, "translate:", "de:", "World"),
        (dummy_npy_file, 0, 0, "translate:", "de:", "Test"),
    ]
    return _write_tsv(
        tmp_path / "features2text.tsv",
        "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput",
        rows,
    )


@pytest.fixture
def signwriting_tsv(tmp_path):
    """TSV for SignWritingDataset."""
    rows = [
        (SIGNWRITING_STRINGS[0], "translate:", "de:", "Hello"),
        (SIGNWRITING_STRINGS[1], "translate:", "de:", "World"),
        (SIGNWRITING_STRINGS[2], "translate:", "de:", "Test"),
    ]
    return _write_tsv(
        tmp_path / "signwriting.tsv",
        "signal\tencoder_prompt\tdecoder_prompt\toutput",
        rows,
    )


@pytest.fixture
def image2text_tsv(tmp_path, dummy_image_file):
    """TSV for BilingualImage2TextDataset."""
    rows = [
        (dummy_image_file, "translate:", "de:", "Hello"),
        (dummy_image_file, "translate:", "de:", "World"),
    ]
    return _write_tsv(
        tmp_path / "image2text.tsv",
        "signal\tencoder_prompt\tdecoder_prompt\toutput",
        rows,
    )


# ---------------------------------------------------------------------------
# Sample batch fixtures  (list-of-dicts for processor / collator tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def text_batch_samples():
    """Batch of samples suitable for Text2TextTranslationProcessor."""
    return [
        {
            "signal": "Hello world",
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hallo Welt",
        },
        {
            "signal": "Good morning",
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Guten Morgen",
        },
    ]


@pytest.fixture
def features_batch_samples(dummy_npy_file):
    """Batch of samples suitable for Features2TextTranslationProcessor."""
    return [
        {
            "signal": dummy_npy_file,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hello",
        },
        {
            "signal": dummy_npy_file,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "World",
        },
    ]


@pytest.fixture
def pose_batch_samples(dummy_pose_file):
    """Batch of samples suitable for Pose2TextTranslationProcessor."""
    return [
        {
            "signal": dummy_pose_file,
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hello",
        },
        {
            "signal": dummy_pose_file,
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "World",
        },
    ]


@pytest.fixture
def video_batch_samples(dummy_video_file):
    """Batch of samples suitable for Video2TextTranslationProcessor."""
    return [
        {
            "signal": dummy_video_file,
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hello",
        },
        {
            "signal": dummy_video_file,
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "World",
        },
    ]


@pytest.fixture
def image_batch_samples(dummy_image_file):
    """Batch of samples suitable for Image2TextTranslationProcessor."""
    return [
        {
            "signal": dummy_image_file,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hello",
        },
        {
            "signal": dummy_image_file,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "World",
        },
    ]


@pytest.fixture
def signwriting_batch_samples():
    """Batch of samples suitable for SignwritingProcessor."""
    return [
        {
            "signal": SIGNWRITING_STRINGS[0],
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hello",
        },
        {
            "signal": SIGNWRITING_STRINGS[1],
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "World",
        },
    ]


# ---------------------------------------------------------------------------
# Asset-based batch samples (use real committed files — for regression tests)
# scope="session" because the asset files never change between tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pose_asset_samples():
    from tests.assets.sample_data import pose_asset_samples as _samples
    return _samples()


@pytest.fixture(scope="session")
def video_asset_samples():
    from tests.assets.sample_data import video_asset_samples as _samples
    return _samples()


@pytest.fixture(scope="session")
def features_asset_samples():
    from tests.assets.sample_data import features_asset_samples as _samples
    return _samples()


@pytest.fixture(scope="session")
def text_asset_samples():
    from tests.assets.sample_data import text_asset_samples as _samples
    return _samples()


@pytest.fixture(scope="session")
def image_asset_samples():
    from tests.assets.sample_data import image_asset_samples as _samples
    return _samples()
