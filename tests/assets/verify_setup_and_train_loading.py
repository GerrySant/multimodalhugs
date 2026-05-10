"""
Verification script: processor setup (multimodalhugs-setup) and train loading
(multimodalhugs-train) path for all supported modalities.

Mirrors exactly what happens in production:

  Setup path  → build_processor_from_config(processor_cfg)   [general_training_setup.py]
                  └─ expand_pipeline_shorthand / slots config
                  └─ save_pretrained to output_dir

  Train path  → import multimodalhugs.processors              [translation_training.py]
                  └─ AutoProcessor.from_pretrained(proc_path)
                  └─ processor.tokenizer

For each modality the script verifies:
  1. build_processor_from_config builds a MultimodalMetaProcessor with no errors
  2. processor.tokenizer is correctly derived from the text slots
  3. AutoProcessor.from_pretrained loads the saved processor back without errors
  4. Loaded processor output matches the output of the freshly built processor
  5. Loaded processor output matches golden values (where a golden file exists)
  6. Loaded processor output matches the corresponding legacy processor

Sample data is imported from tests/assets/sample_data.py — the same source used by
tests/test_data/conftest.py fixtures and tests/assets/generate_golden.py.

Usage
-----
    # from the repo root, with the mmhugs conda env active:
    python tests/assets/verify_setup_and_train_loading.py

    # or via conda run:
    conda run -n mmhugs python tests/assets/verify_setup_and_train_loading.py
"""

import json
import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf

# ── Sample data and paths — single source of truth ───────────────────────────
from tests.assets.sample_data import (
    ASSETS_DIR,
    TINY_TOKENIZER_PATH,
    FONT_PATH,
    CLIP_PROCESSOR_PATH,
    pose_asset_samples,
    video_asset_samples,
    features_asset_samples,
    text_asset_samples,
    image_asset_samples,
    signwriting_asset_samples,
)

GOLDEN_DIR = os.path.join(ASSETS_DIR, "golden")

# ── Setup path import (same as general_training_setup.py) ────────────────────
from multimodalhugs.training_setup.setup_utils import build_processor_from_config

# ── Train loading path import (same as translation_training.py) ──────────────
import multimodalhugs.processors  # triggers AutoProcessor registration for all classes
from transformers import AutoProcessor

# ── Legacy processors for parity checks ──────────────────────────────────────
from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor
from multimodalhugs.processors.legacy.pose2text_preprocessor import Pose2TextTranslationProcessor
from multimodalhugs.processors.legacy.video2text_preprocessor import Video2TextTranslationProcessor
from multimodalhugs.processors.legacy.features2text_preprocessor import Features2TextTranslationProcessor
from multimodalhugs.processors.legacy.image2text_preprocessor import Image2TextTranslationProcessor
from multimodalhugs.processors.legacy.signwriting_preprocessor import SignwritingProcessor
from multimodalhugs.processors.legacy.text2text_preprocessor import Text2TextTranslationProcessor
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast.from_pretrained(TINY_TOKENIZER_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

ABS_TOL = 1e-4


def load_golden(name):
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def check_vs_golden(result, golden, label):
    abs_tol = golden.get("abs_tol", ABS_TOL)
    for key, g in golden.items():
        if key == "abs_tol":
            continue
        t = result[key]
        assert list(t.shape) == g["shape"], (
            f"[{label}] {key} shape mismatch: got {list(t.shape)}, expected {g['shape']}"
        )
        assert str(t.dtype) == g["dtype"], f"[{label}] {key} dtype mismatch"
        if t.is_floating_point():
            f = t.float()
            assert abs(f.mean().item() - g["mean"]) < abs_tol, f"[{label}] {key} mean mismatch"
            assert abs(f.sum().item() - g["sum"]) < abs_tol * t.numel(), f"[{label}] {key} sum mismatch"
        else:
            assert int(t.sum().item()) == g["sum"], f"[{label}] {key} sum mismatch"
            assert t.tolist() == g["values"], f"[{label}] {key} values mismatch"


def check_identical(a, b, label):
    assert set(a.keys()) == set(b.keys()), (
        f"[{label}] key mismatch: {sorted(a.keys())} vs {sorted(b.keys())}"
    )
    for key in a:
        ta, tb = a[key], b[key]
        if ta.is_floating_point():
            assert torch.allclose(ta, tb, atol=ABS_TOL), f"[{label}] {key} not close"
        else:
            assert ta.tolist() == tb.tolist(), f"[{label}] {key} values differ"


def _text_slot_cfgs(tokenizer_path):
    """Three standard text output slots in YAML-dict format (used by build_processor_from_config)."""
    return [
        {
            "processor_class": "TextModalityProcessor",
            "processor_kwargs": {"tokenizer_path": tokenizer_path, "role": "target"},
            "output_data_key": "labels",
            "is_label": True,
            "column_map": {"decoder_prompt": "target_prefix", "output": "target"},
        },
        {
            "processor_class": "TextModalityProcessor",
            "processor_kwargs": {"tokenizer_path": tokenizer_path, "role": "input"},
            "output_data_key": "encoder_prompt",
            "output_mask_key": "encoder_prompt_length_padding_mask",
            "column_map": {"encoder_prompt": "signal"},
        },
        {
            "processor_class": "TextModalityProcessor",
            "processor_kwargs": {"tokenizer_path": tokenizer_path, "role": "input"},
            "output_data_key": "decoder_input_ids",
            "output_mask_key": "decoder_attention_mask",
            "column_map": {"decoder_prompt": "signal"},
        },
    ]


def run(name, processor_cfg_dict, samples, legacy, golden_name=None):
    """
    Full setup + train-loading verification for one modality.

    processor_cfg_dict  — the 'processor:' section of a config YAML as a dict.
    samples             — list-of-dicts batch to call the processor with.
    legacy              — instantiated legacy processor for parity check.
    golden_name         — key for the golden JSON file (defaults to name).
    """
    processor_cfg = OmegaConf.create(processor_cfg_dict)

    # ── 1. Setup path: build_processor_from_config ───────────────────────────
    proc = build_processor_from_config(processor_cfg)
    assert proc is not None, f"[{name}] build_processor_from_config returned None"
    assert isinstance(proc, MultimodalMetaProcessor), (
        f"[{name}] Expected MultimodalMetaProcessor, got {type(proc).__name__}"
    )
    print(f"  [setup]   build_processor_from_config → {type(proc).__name__}, "
          f"slots={len(proc.slots)}")

    # ── 2. Tokenizer derived from slots ──────────────────────────────────────
    assert proc.tokenizer is not None, f"[{name}] proc.tokenizer should not be None"
    print(f"  [setup]   proc.tokenizer = {type(proc.tokenizer).__name__}  "
          f"(derived from slot @property)")

    # ── 3. Call freshly built processor ──────────────────────────────────────
    out_built = proc(samples)
    print(f"  [setup]   output keys: {sorted(out_built.keys())}")

    with tempfile.TemporaryDirectory() as tmp:
        proc.save_pretrained(tmp)
        print(f"  [setup]   saved to {tmp}")

        # ── 4. Train loading path: AutoProcessor.from_pretrained ─────────────
        loaded = AutoProcessor.from_pretrained(tmp)
        assert isinstance(loaded, MultimodalMetaProcessor), (
            f"[{name}] AutoProcessor.from_pretrained returned {type(loaded).__name__}, "
            f"expected MultimodalMetaProcessor"
        )
        assert loaded.tokenizer is not None, (
            f"[{name}] loaded.tokenizer is None after AutoProcessor.from_pretrained"
        )
        print(f"  [train]   AutoProcessor.from_pretrained → {type(loaded).__name__}, "
              f"slots={len(loaded.slots)}")
        print(f"  [train]   loaded.tokenizer = {type(loaded.tokenizer).__name__}")

        # ── 5. Loaded output == built output ─────────────────────────────────
        out_loaded = loaded(samples)
        check_identical(out_built, out_loaded, f"{name}: built vs loaded")
        print(f"  [check]   Built == Loaded             : PASSED")

        # ── 6. Golden ─────────────────────────────────────────────────────────
        golden = load_golden(golden_name or name)
        if golden:
            check_vs_golden(out_loaded, golden, name)
            print(f"  [check]   Golden                      : PASSED")
        else:
            print(f"  [check]   Golden                      : SKIPPED (no golden file)")

        # ── 7. Legacy parity ─────────────────────────────────────────────────
        out_legacy = legacy(samples)
        check_identical(out_loaded, out_legacy, f"{name}: loaded vs legacy")
        print(f"  [check]   Legacy parity               : PASSED")


# ── Processor slot configs (mirrors what users write under 'processor:' in YAML) ─
OFFSET_COL = {"signal": "signal", "signal_start": "signal_start", "signal_end": "signal_end"}

text_cfg = {
    "slots": [
        {
            "processor_class": "TextModalityProcessor",
            "processor_kwargs": {"tokenizer_path": TINY_TOKENIZER_PATH, "role": "input"},
            "output_data_key": "input_ids",
            "output_mask_key": "attention_mask",
            "column_map": {"signal": "signal"},
        },
        *_text_slot_cfgs(TINY_TOKENIZER_PATH),
    ]
}

pose_cfg = {
    "slots": [
        {
            "processor_class": "PoseModalityProcessor",
            "processor_kwargs": {"reduce_holistic_poses": True},
            "output_data_key": "input_frames",
            "output_mask_key": "attention_mask",
            "column_map": OFFSET_COL,
        },
        *_text_slot_cfgs(TINY_TOKENIZER_PATH),
    ]
}

video_cfg = {
    "slots": [
        {
            "processor_class": "VideoModalityProcessor",
            "processor_kwargs": {"custom_preprocessor_path": CLIP_PROCESSOR_PATH, "use_cache": True},
            "output_data_key": "input_frames",
            "output_mask_key": "attention_mask",
            "column_map": OFFSET_COL,
        },
        *_text_slot_cfgs(TINY_TOKENIZER_PATH),
    ]
}

features_cfg = {
    "slots": [
        # FeaturesModalityProcessor uses only the 'signal' column (file path);
        # offsets are not passed via column_map — the processor handles them internally.
        {
            "processor_class": "FeaturesModalityProcessor",
            "processor_kwargs": {"use_cache": False},
            "output_data_key": "input_frames",
            "output_mask_key": "attention_mask",
        },
        *_text_slot_cfgs(TINY_TOKENIZER_PATH),
    ]
}

image_cfg = {
    "slots": [
        {
            "processor_class": "ImageModalityProcessor",
            "processor_kwargs": {
                "font_path": FONT_PATH,
                "width": 224,
                "height": 224,
                "normalize_image": False,
            },
            "output_data_key": "input_frames",
            "output_mask_key": "attention_mask",
            "column_map": {"signal": "signal"},
        },
        *_text_slot_cfgs(TINY_TOKENIZER_PATH),
    ]
}

signwriting_cfg = {
    "slots": [
        {
            "processor_class": "SignwritingModalityProcessor",
            "processor_kwargs": {"custom_preprocessor_path": CLIP_PROCESSOR_PATH},
            "output_data_key": "input_frames",
            "output_mask_key": "attention_mask",
            "column_map": {"signal": "signal"},
        },
        *_text_slot_cfgs(TINY_TOKENIZER_PATH),
    ]
}

# ── Run all modalities ─────────────────────────────────────────────────────────
SEPARATOR = "=" * 60
errors = []


def section(title, fn):
    print(SEPARATOR)
    print(title)
    try:
        fn()
    except AssertionError as e:
        print(f"  FAILED : {e}")
        errors.append((title, str(e)))
    except Exception as e:
        print(f"  ERROR  : {type(e).__name__}: {e}")
        errors.append((title, f"{type(e).__name__}: {e}"))
    print()


section("TEXT only (text → text)", lambda: run(
    "text2text", text_cfg, text_asset_samples(),
    Text2TextTranslationProcessor(tokenizer=tok),
))

section("POSE + Text", lambda: run(
    "pose2text", pose_cfg, pose_asset_samples(),
    Pose2TextTranslationProcessor(tokenizer=tok, reduce_holistic_poses=True),
))

section("VIDEO + Text", lambda: run(
    "video2text", video_cfg, video_asset_samples(),
    Video2TextTranslationProcessor(tokenizer=tok, custom_preprocessor_path=CLIP_PROCESSOR_PATH, use_cache=True),
))

section("FEATURES + Text", lambda: run(
    "features2text", features_cfg, features_asset_samples(),
    Features2TextTranslationProcessor(tokenizer=tok, use_cache=False),
))

section("IMAGE + Text", lambda: run(
    "image2text", image_cfg, image_asset_samples(),
    Image2TextTranslationProcessor(
        tokenizer=tok, font_path=FONT_PATH, width=224, height=224, normalize_image=False
    ),
))

section("SIGNWRITING + Text", lambda: run(
    "signwriting2text", signwriting_cfg, signwriting_asset_samples(),
    SignwritingProcessor(tokenizer=tok, custom_preprocessor_path=CLIP_PROCESSOR_PATH),
    golden_name=None,
))

# ── Summary ────────────────────────────────────────────────────────────────────
print(SEPARATOR)
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for title, msg in errors:
        print(f"  [{title}] {msg}")
    sys.exit(1)
else:
    print("ALL MODALITY COMBINATIONS: PASSED")
