"""
Canonical sample data for multimodalhugs tests and verification scripts.

This module is the single source of truth for:
  - Path constants (assets, tokenizer, processors, fonts)
  - Asset-based sample batches used in regression tests and golden-file generation

All of the following must stay byte-for-byte consistent with this module:
  - tests/test_data/conftest.py   (pytest fixtures that return these samples)
  - tests/assets/generate_golden.py  (generates the committed golden JSON files)
  - tests/assets/verify_meta_processor.py
  - tests/assets/verify_setup_and_train_loading.py

SIGNWRITING_STRINGS are also imported by conftest.py for non-asset (batch) fixtures.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ASSETS_DIR         = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR          = os.path.dirname(_ASSETS_DIR)
_REPO_ROOT          = os.path.dirname(_TESTS_DIR)

ASSETS_DIR          = _ASSETS_DIR
TINY_TOKENIZER_PATH = os.path.join(_TESTS_DIR, "test_model_only", "tiny_tokenizer")
FONT_PATH           = os.path.join(_TESTS_DIR, "e2e_overfitting", "other_files", "Arial.ttf")
CLIP_PROCESSOR_PATH = os.path.join(_ASSETS_DIR, "processors", "clip_image_processor")

# ---------------------------------------------------------------------------
# SignWriting FSW strings  (shared with batch fixtures in conftest.py)
# ---------------------------------------------------------------------------
SIGNWRITING_STRINGS = [
    "M518x529S14c20481x471S27106503x489",
    "M522x525S11541498x491S11549479x498",
    "M524x518S15a28476x483S20e00499x489",
]

# ---------------------------------------------------------------------------
# Asset-based sample batches
# These use real committed binary files (pose, video, npy) or inline strings
# (text, image, signwriting).  The values match the golden files in
# tests/assets/golden/ and the conftest.py *_asset_samples fixtures exactly.
# ---------------------------------------------------------------------------

def pose_asset_samples():
    pose_dir = os.path.join(ASSETS_DIR, "pose")
    return [
        {
            "signal": os.path.join(pose_dir, "sample_01.pose"),
            "signal_start": 0, "signal_end": 0,
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Let's open Access.",
        },
        {
            "signal": os.path.join(pose_dir, "sample_02.pose"),
            "signal_start": 0, "signal_end": 0,
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Good.",
        },
    ]


def video_asset_samples():
    video_dir = os.path.join(ASSETS_DIR, "video")
    return [
        {
            "signal": os.path.join(video_dir, "sample_01.mp4"),
            "signal_start": 0, "signal_end": 0,
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Let's open Access.",
        },
        {
            "signal": os.path.join(video_dir, "sample_02.mp4"),
            "signal_start": 0, "signal_end": 0,
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Good.",
        },
    ]


def features_asset_samples():
    features_dir = os.path.join(ASSETS_DIR, "features")
    return [
        {
            "signal": os.path.join(features_dir, "sample_01.npy"),
            "signal_start": 0, "signal_end": 0,
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Let's open Access.",
        },
        {
            "signal": os.path.join(features_dir, "sample_02.npy"),
            "signal_start": 0, "signal_end": 0,
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Good.",
        },
    ]


def text_asset_samples():
    return [
        {
            "signal": "Let's open Access.",
            "encoder_prompt": "__en__", "decoder_prompt": "__fr__",
            "output": "Ouvrons Access.",
        },
        {
            "signal": "Good.",
            "encoder_prompt": "__en__", "decoder_prompt": "__fr__",
            "output": "Bien.",
        },
    ]


def image_asset_samples():
    # ImageModalityProcessor renders the signal string as an image.
    return [
        {
            "signal": "Let's open Access.",
            "encoder_prompt": "lowercase: ", "decoder_prompt": "__en__",
            "output": "let's open access.",
        },
        {
            "signal": "Good.",
            "encoder_prompt": "lowercase: ", "decoder_prompt": "__en__",
            "output": "good.",
        },
    ]


def signwriting_asset_samples():
    # Uses the real committed FSW strings from the signwriting metadata TSV.
    # No golden file exists for signwriting; legacy parity is verified instead.
    return [
        {
            "signal": SIGNWRITING_STRINGS[0],
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Let's open Access.",
        },
        {
            "signal": SIGNWRITING_STRINGS[1],
            "encoder_prompt": "__asl__", "decoder_prompt": "__en__",
            "output": "Good.",
        },
    ]
