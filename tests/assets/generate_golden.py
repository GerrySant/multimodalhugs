"""
Golden-value generator for processor regression tests.

Run this script once (and again after any intentional preprocessing change)
to capture the current pipeline outputs as a stable reference.  The golden
files are committed to git so that future runs can detect unintended changes.

Usage:
    python tests/assets/generate_golden.py

Output:
    tests/assets/golden/<modality>.json  — one file per modality

IMPORTANT — keep batch definitions in sync with conftest fixtures
-----------------------------------------------------------------
The sample dicts defined in each generate_*() function below must stay
in sync with the corresponding asset-based fixtures in
tests/test_data/conftest.py (pose_asset_samples, video_asset_samples, etc.).

If you change the signal paths, encoder_prompt, decoder_prompt, or output
values here, update the matching fixture in conftest.py as well, and
regenerate the golden files.  A mismatch between the two will cause
regression tests to fail with unexpected key/value errors.
"""

import json
import os
import sys

import torch
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ASSETS_DIR  = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR   = os.path.dirname(ASSETS_DIR)
GOLDEN_DIR  = os.path.join(ASSETS_DIR, "golden")
TINY_TOKENIZER_PATH = os.path.join(TESTS_DIR, "test_model_only", "tiny_tokenizer")
FONT_PATH   = os.path.join(TESTS_DIR, "e2e_overfitting", "other_files", "Arial.ttf")

# Make sure the package is importable when running from the repo root
sys.path.insert(0, os.path.dirname(TESTS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokenizer():
    tok = PreTrainedTokenizerFast.from_pretrained(TINY_TOKENIZER_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tensor_summary(t: torch.Tensor, fingerprint: bool = True) -> dict:
    """
    Capture enough information about a tensor to detect any change in future runs.

    For floating-point tensors (input_frames, masks stored as float):
        shape, dtype, mean, std, min, max, sum
        If fingerprint=True, also stores first_values and last_values (8 elements
        each from the flattened tensor) to catch permutations or transpositions.
        Set fingerprint=False for processors that rely on platform-specific
        libraries (e.g. mediapipe) where exact element values may differ across
        hardware architectures.
    For integer tensors (token ids, integer masks):
        shape, dtype, sum, and the full values list (they are small).
    """
    summary = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
    }
    if t.is_floating_point():
        f = t.float()
        summary.update({
            "mean": round(f.mean().item(), 8),
            "std":  round(f.std().item(),  8) if f.numel() > 1 else 0.0,
            "min":  round(f.min().item(),  8),
            "max":  round(f.max().item(),  8),
            "sum":  round(f.sum().item(),  6),
        })
        if fingerprint:
            flat = f.flatten()
            summary.update({
                "first_values": [round(v, 6) for v in flat[:8].tolist()],
                "last_values":  [round(v, 6) for v in flat[-8:].tolist()],
            })
    else:
        summary.update({
            "sum":    int(t.sum().item()),
            "values": t.tolist(),
        })
    return summary


def capture(result, fingerprint: bool = True) -> dict:
    """Summarise all tensor values in result into a JSON-serialisable dict."""
    return {
        key: tensor_summary(val, fingerprint=fingerprint)
        for key, val in result.items()
        if isinstance(val, torch.Tensor)
    }


def save(name: str, data: dict):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Per-modality generators
# ---------------------------------------------------------------------------

def generate_pose():
    from multimodalhugs.processors.pose2text_preprocessor import Pose2TextTranslationProcessor
    pose_dir = os.path.join(ASSETS_DIR, "pose")
    batch = [
        {"signal": os.path.join(pose_dir, "sample_01.pose"), "signal_start": 0, "signal_end": 0,
         "encoder_prompt": "__asl__", "decoder_prompt": "__en__", "output": "Let's open Access."},
        {"signal": os.path.join(pose_dir, "sample_02.pose"), "signal_start": 0, "signal_end": 0,
         "encoder_prompt": "__asl__", "decoder_prompt": "__en__", "output": "Good."},
    ]
    processor = Pose2TextTranslationProcessor(tokenizer=load_tokenizer(), reduce_holistic_poses=True)
    save("pose2text", capture(processor(batch=batch), fingerprint=False))


def generate_video():
    # Uses a local CLIPImageProcessor (saved by generate_assets.py) to resize
    # all frames to 224×224 before batching, so both samples can be collated
    # together regardless of their original resolution.
    from multimodalhugs.processors.video2text_preprocessor import Video2TextTranslationProcessor
    clip_processor_path = os.path.join(ASSETS_DIR, "processors", "clip_image_processor")
    if not os.path.exists(clip_processor_path):
        print("  CLIP processor not found — run generate_assets.py first.  Skipping video.")
        return
    video_dir = os.path.join(ASSETS_DIR, "video")
    batch = [
        {"signal": os.path.join(video_dir, "sample_01.mp4"), "signal_start": 0, "signal_end": 0,
         "encoder_prompt": "__asl__", "decoder_prompt": "__en__", "output": "Let's open Access."},
        {"signal": os.path.join(video_dir, "sample_02.mp4"), "signal_start": 0, "signal_end": 0,
         "encoder_prompt": "__asl__", "decoder_prompt": "__en__", "output": "Good."},
    ]
    processor = Video2TextTranslationProcessor(
        tokenizer=load_tokenizer(),
        custom_preprocessor_path=clip_processor_path,
    )
    data = capture(processor(batch=batch), fingerprint=False)
    # PyAV decodes pixel values slightly differently across OS/codec versions.
    # Store a relaxed per-golden tolerance so the test tolerates platform drift.
    data["abs_tol"] = 0.01
    save("video2text", data)


def generate_features():
    from multimodalhugs.processors.features2text_preprocessor import Features2TextTranslationProcessor
    features_dir = os.path.join(ASSETS_DIR, "features")
    batch = [
        {"signal": os.path.join(features_dir, "sample_01.npy"), "signal_start": 0, "signal_end": 0,
         "encoder_prompt": "__asl__", "decoder_prompt": "__en__", "output": "Let's open Access."},
        {"signal": os.path.join(features_dir, "sample_02.npy"), "signal_start": 0, "signal_end": 0,
         "encoder_prompt": "__asl__", "decoder_prompt": "__en__", "output": "Good."},
    ]
    processor = Features2TextTranslationProcessor(tokenizer=load_tokenizer(), use_cache=False)
    save("features2text", capture(processor(batch=batch)))


def generate_text():
    from multimodalhugs.processors.text2text_preprocessor import Text2TextTranslationProcessor
    batch = [
        {"signal": "Let's open Access.", "encoder_prompt": "__en__",
         "decoder_prompt": "__fr__", "output": "Ouvrons Access."},
        {"signal": "Good.", "encoder_prompt": "__en__",
         "decoder_prompt": "__fr__", "output": "Bien."},
    ]
    processor = Text2TextTranslationProcessor(tokenizer=load_tokenizer())
    save("text2text", capture(processor(batch=batch)))


def generate_labels():
    """
    Capture the output of create_seq2seq_labels_from_samples with the text asset samples.

    This function will be replaced by TextModalityProcessor(role='label') during the
    processor refactoring; the golden file defines the contract the new implementation must meet.
    """
    from multimodalhugs.data.datacollators.multimodal_datacollator import (
        create_seq2seq_labels_from_samples,
    )
    # Same text samples used by generate_text(), same tokenizer
    batch = [
        {"signal": "Let's open Access.", "encoder_prompt": "__en__",
         "decoder_prompt": "__fr__", "output": "Ouvrons Access."},
        {"signal": "Good.", "encoder_prompt": "__en__",
         "decoder_prompt": "__fr__", "output": "Bien."},
    ]
    result = create_seq2seq_labels_from_samples(
        samples=batch,
        tokenizer=load_tokenizer(),
        label_pad_token_id=-100,
        padding=True,
        return_tensors="pt",
    )
    save("labels", capture(result))


def generate_image():
    from multimodalhugs.processors.image2text_preprocessor import Image2TextTranslationProcessor
    batch = [
        {"signal": "Let's open Access.", "encoder_prompt": "lowercase: ",
         "decoder_prompt": "__en__", "output": "let's open access."},
        {"signal": "Good.", "encoder_prompt": "lowercase: ",
         "decoder_prompt": "__en__", "output": "good."},
    ]
    processor = Image2TextTranslationProcessor(
        tokenizer=load_tokenizer(), font_path=FONT_PATH,
        width=224, height=224, normalize_image=False,
    )
    save("image2text", capture(processor(batch=batch)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = {
    "pose":     generate_pose,
    "video":    generate_video,
    "features": generate_features,
    "text":     generate_text,
    "labels":   generate_labels,
    "image":    generate_image,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(GENERATORS)
    print("Generating golden files...")
    for name in targets:
        if name not in GENERATORS:
            print(f"  Unknown modality '{name}'. Choose from: {list(GENERATORS)}")
            sys.exit(1)
        print(f"  [{name}]")
        GENERATORS[name]()
    print("Done.")
