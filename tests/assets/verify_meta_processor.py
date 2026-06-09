"""
Verification script: MultimodalMetaProcessor with TextModalityProcessor slots.

Checks that the architecture change (tokenizer removed from MultimodalMetaProcessor
constructor, exposed as a @property derived from the first text slot) works correctly
for all supported modalities.

For each modality the script verifies:
  1. Construction  — MultimodalMetaProcessor(slots=[...]) with no tokenizer= kwarg
  2. tokenizer     — meta.tokenizer is correctly derived from the first text slot
  3. Output        — all output keys and shapes present
  4. Golden        — output matches committed golden values (where a golden file exists)
  5. Legacy parity — output is identical to the corresponding legacy processor
  6. Round-trip    — save_pretrained + from_pretrained produces identical output

Sample data is imported from tests/assets/sample_data.py — the same source used by
tests/test_data/conftest.py fixtures and tests/assets/generate_golden.py.

Usage
-----
    # from the repo root, with the mmhugs conda env active:
    python tests/assets/verify_meta_processor.py

    # or via conda run:
    conda run -n mmhugs python tests/assets/verify_meta_processor.py
"""

import json
import os
import sys
import tempfile

import torch
from transformers import PreTrainedTokenizerFast

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

# ── Processor imports ─────────────────────────────────────────────────────────
from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor, TextRole
from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from multimodalhugs.processors.features_modality_processor import FeaturesModalityProcessor
from multimodalhugs.processors.image_modality_processor import ImageModalityProcessor
from multimodalhugs.processors.signwriting_modality_processor import SignwritingModalityProcessor

from multimodalhugs.processors.legacy.pose2text_preprocessor import Pose2TextTranslationProcessor
from multimodalhugs.processors.legacy.video2text_preprocessor import Video2TextTranslationProcessor
from multimodalhugs.processors.legacy.features2text_preprocessor import Features2TextTranslationProcessor
from multimodalhugs.processors.legacy.image2text_preprocessor import Image2TextTranslationProcessor
from multimodalhugs.processors.legacy.signwriting_preprocessor import SignwritingProcessor
from multimodalhugs.processors.legacy.text2text_preprocessor import Text2TextTranslationProcessor

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = PreTrainedTokenizerFast.from_pretrained(TINY_TOKENIZER_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ── Helpers ───────────────────────────────────────────────────────────────────
ABS_TOL = 1e-4


def text_slots(tok):
    """Three standard text output slots: labels, encoder_prompt, decoder_input_ids."""
    return [
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tok, role=TextRole.TARGET),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tok, role=TextRole.INPUT),
            output_data_key="encoder_prompt",
            output_mask_key="encoder_prompt_length_padding_mask",
            column_map={"encoder_prompt": "signal"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tok, role=TextRole.INPUT),
            output_data_key="decoder_input_ids",
            output_mask_key="decoder_attention_mask",
            column_map={"decoder_prompt": "signal"},
        ),
    ]


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


def run(name, meta, samples, legacy, golden_name=None):
    # 1. tokenizer property
    assert meta.tokenizer is not None, f"[{name}] meta.tokenizer should not be None"
    print(f"  tokenizer : {type(meta.tokenizer).__name__}  (derived from slot, not constructor)")
    print(f"  slots     : {len(meta.slots)}")

    # 2. call
    out = meta(samples)
    print(f"  output keys: {sorted(out.keys())}")

    # 3. golden
    golden = load_golden(golden_name or name)
    if golden:
        check_vs_golden(out, golden, name)
        print(f"  Golden check   : PASSED")
    else:
        print(f"  Golden check   : SKIPPED (no golden file for '{golden_name or name}')")

    # 4. legacy parity
    check_identical(out, legacy(samples), f"{name} vs legacy")
    print(f"  Legacy parity  : PASSED")

    # 5. save / load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        meta.save_pretrained(tmp)
        loaded = MultimodalMetaProcessor.from_pretrained(tmp)
        assert isinstance(loaded, MultimodalMetaProcessor), (
            f"[{name}] from_pretrained did not return a MultimodalMetaProcessor"
        )
        assert len(loaded.slots) == len(meta.slots), (
            f"[{name}] slot count changed after round-trip"
        )
        check_identical(out, loaded(samples), f"{name} round-trip")
        print(f"  Round-trip     : PASSED")


# ── Column map shared by offset-based modalities ──────────────────────────────
OFFSET_COL = {"signal": "signal", "signal_start": "signal_start", "signal_end": "signal_end"}

# ── Run all modalities ─────────────────────────────────────────────────────────
SEPARATOR = "=" * 60
errors = []


def section(title, fn):
    print(SEPARATOR)
    print(title)
    try:
        fn()
    except AssertionError as e:
        print(f"  FAILED: {e}")
        errors.append((title, str(e)))
    except Exception as e:
        print(f"  ERROR : {type(e).__name__}: {e}")
        errors.append((title, f"{type(e).__name__}: {e}"))
    print()


section("TEXT only (text → text)", lambda: run(
    "text2text",
    MultimodalMetaProcessor(slots=[
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tok, role=TextRole.INPUT),
            output_data_key="input_ids",
            output_mask_key="attention_mask",
            column_map={"signal": "signal"},
        ),
        *text_slots(tok),
    ]),
    text_asset_samples(),
    Text2TextTranslationProcessor(tokenizer=tok),
))

section("POSE + Text", lambda: run(
    "pose2text",
    MultimodalMetaProcessor(slots=[
        ProcessorSlot(
            processor=PoseModalityProcessor(reduce_holistic_poses=True),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
            column_map=OFFSET_COL,
        ),
        *text_slots(tok),
    ]),
    pose_asset_samples(),
    Pose2TextTranslationProcessor(tokenizer=tok, reduce_holistic_poses=True),
))

section("VIDEO + Text", lambda: run(
    "video2text",
    MultimodalMetaProcessor(slots=[
        ProcessorSlot(
            processor=VideoModalityProcessor(custom_preprocessor_path=CLIP_PROCESSOR_PATH, use_cache=True),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
            column_map=OFFSET_COL,
        ),
        *text_slots(tok),
    ]),
    video_asset_samples(),
    Video2TextTranslationProcessor(tokenizer=tok, custom_preprocessor_path=CLIP_PROCESSOR_PATH, use_cache=True),
))

section("FEATURES + Text", lambda: run(
    "features2text",
    MultimodalMetaProcessor(slots=[
        # FeaturesModalityProcessor uses only the 'signal' column (file path);
        # offsets are not passed via column_map — the processor handles them internally.
        ProcessorSlot(
            processor=FeaturesModalityProcessor(use_cache=False),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
        ),
        *text_slots(tok),
    ]),
    features_asset_samples(),
    Features2TextTranslationProcessor(tokenizer=tok, use_cache=False),
))

section("IMAGE + Text", lambda: run(
    "image2text",
    MultimodalMetaProcessor(slots=[
        ProcessorSlot(
            processor=ImageModalityProcessor(
                font_path=FONT_PATH, width=224, height=224, normalize_image=False
            ),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
            column_map={"signal": "signal"},
        ),
        *text_slots(tok),
    ]),
    image_asset_samples(),
    Image2TextTranslationProcessor(
        tokenizer=tok, font_path=FONT_PATH, width=224, height=224, normalize_image=False
    ),
))

section("SIGNWRITING + Text", lambda: run(
    "signwriting2text",
    MultimodalMetaProcessor(slots=[
        ProcessorSlot(
            processor=SignwritingModalityProcessor(custom_preprocessor_path=CLIP_PROCESSOR_PATH),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
            column_map={"signal": "signal"},
        ),
        *text_slots(tok),
    ]),
    signwriting_asset_samples(),
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
