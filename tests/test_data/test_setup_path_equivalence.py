"""
Equivalence tests: --modality (legacy) path vs general path (dataset_type in config).

For every supported modality both paths must produce an identical
processor_config.json.  The test uses the committed binary assets under
tests/assets/ so no internet access or temporary file generation is needed.
"""

import json
import os

import pytest
from omegaconf import OmegaConf

from tests.test_data.conftest import (
    ASSETS_DIR,
    CLIP_PROCESSOR_PATH,
    FONT_PATH,
    TINY_TOKENIZER_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_yaml(path, data: dict):
    OmegaConf.save(OmegaConf.create(data), str(path))


def _run_general(config_path, output_dir):
    """Run via the general path (no --modality, dataset_type in config)."""
    from multimodalhugs.training_setup.general_training_setup import main
    main(
        config_path=str(config_path),
        do_dataset=False,
        do_processor=True,
        do_model=False,
        output_dir=str(output_dir),
    )
    # resolve_setup_paths appends "setup/"; processor saves to
    # "{output_dir}/setup/multimodal_meta_processor"
    return output_dir / "setup" / "multimodal_meta_processor"


def _run_legacy(modality, config_path, output_dir):
    """Run via the legacy --modality path."""
    from importlib import import_module
    module = import_module(
        f"multimodalhugs.training_setup.{modality}_training_setup"
    )
    module.main(
        config_path=str(config_path),
        do_dataset=False,
        do_processor=True,
        do_model=False,
        output_dir=str(output_dir),
    )
    return output_dir / "setup" / "multimodal_meta_processor"


def _load_processor_dict(proc_dir):
    with open(proc_dir / "processor_config.json") as f:
        return json.load(f)


def _assert_processors_equivalent(proc_a: dict, proc_b: dict):
    """Assert two processor dicts have identical slot structure."""
    assert proc_a["processor_class"] == proc_b["processor_class"]
    slots_a, slots_b = proc_a["slots"], proc_b["slots"]
    assert len(slots_a) == len(slots_b), (
        f"Slot count: general={len(slots_a)}, legacy={len(slots_b)}"
    )
    for i, (sa, sb) in enumerate(zip(slots_a, slots_b)):
        assert sa["processor_class"] == sb["processor_class"], f"slot {i}: processor_class"
        assert sa.get("output_data_key") == sb.get("output_data_key"), f"slot {i}: output_data_key"
        assert sa.get("output_mask_key") == sb.get("output_mask_key"), f"slot {i}: output_mask_key"
        assert sa.get("is_label", False) == sb.get("is_label", False), f"slot {i}: is_label"
        assert sa.get("column_map") == sb.get("column_map"), f"slot {i}: column_map"


# ---------------------------------------------------------------------------
# Per-modality config specs
# ---------------------------------------------------------------------------
# Each entry: (dataset_type, legacy_modality_name, processor_cfg, data_cfg)

_META = os.path.join(ASSETS_DIR, "{modality}", "metadata.tsv")

MODALITY_SPECS = [
    (
        "pose2text",
        "pose2text",
        {
            "pipeline": "pose2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        },
        {
            "train_metadata_file": os.path.join(ASSETS_DIR, "pose", "metadata.tsv"),
            "validation_metadata_file": os.path.join(ASSETS_DIR, "pose", "metadata.tsv"),
            "test_metadata_file": os.path.join(ASSETS_DIR, "pose", "metadata.tsv"),
        },
    ),
    (
        "video2text",
        "video2text",
        {
            "pipeline": "video2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        },
        {
            "train_metadata_file": os.path.join(ASSETS_DIR, "video", "metadata.tsv"),
            "validation_metadata_file": os.path.join(ASSETS_DIR, "video", "metadata.tsv"),
            "test_metadata_file": os.path.join(ASSETS_DIR, "video", "metadata.tsv"),
        },
    ),
    (
        "features2text",
        "features2text",
        {
            "pipeline": "features2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        },
        {
            "train_metadata_file": os.path.join(ASSETS_DIR, "features", "metadata.tsv"),
            "validation_metadata_file": os.path.join(ASSETS_DIR, "features", "metadata.tsv"),
            "test_metadata_file": os.path.join(ASSETS_DIR, "features", "metadata.tsv"),
        },
    ),
    (
        "signwriting2text",
        "signwriting2text",
        {
            "pipeline": "signwriting2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
            "modality_kwargs": {
                "custom_preprocessor_path": CLIP_PROCESSOR_PATH,
            },
        },
        {
            "train_metadata_file": os.path.join(ASSETS_DIR, "signwriting", "metadata.tsv"),
            "validation_metadata_file": os.path.join(ASSETS_DIR, "signwriting", "metadata.tsv"),
            "test_metadata_file": os.path.join(ASSETS_DIR, "signwriting", "metadata.tsv"),
        },
    ),
    (
        "image2text",
        "image2text",
        {
            "pipeline": "image2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
            "modality_kwargs": {
                "font_path": FONT_PATH,
                "width": 64,
                "height": 64,
                "normalize_image": False,
            },
        },
        {
            "train_metadata_file": os.path.join(ASSETS_DIR, "image", "metadata.tsv"),
            "validation_metadata_file": os.path.join(ASSETS_DIR, "image", "metadata.tsv"),
            "test_metadata_file": os.path.join(ASSETS_DIR, "image", "metadata.tsv"),
        },
    ),
    (
        "text2text",
        "text2text",
        {
            "pipeline": "text2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        },
        {
            "train_metadata_file": os.path.join(ASSETS_DIR, "text", "metadata.tsv"),
            "validation_metadata_file": os.path.join(ASSETS_DIR, "text", "metadata.tsv"),
            "test_metadata_file": os.path.join(ASSETS_DIR, "text", "metadata.tsv"),
        },
    ),
]

MODALITY_IDS = [spec[0] for spec in MODALITY_SPECS]


# ---------------------------------------------------------------------------
# Parametrized equivalence tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_type,legacy_modality,processor_cfg,data_cfg", MODALITY_SPECS, ids=MODALITY_IDS)
class TestSetupPathEquivalence:
    """General path vs --modality legacy path must produce identical processor artifacts."""

    def test_processor_slot_structure_identical(
        self, dataset_type, legacy_modality, processor_cfg, data_cfg, tmp_path
    ):
        # general path: dataset_type in config
        cfg_general = {
            "data": {"dataset_type": dataset_type, **data_cfg},
            "processor": processor_cfg,
            "setup": {"output_dir": str(tmp_path / "general")},
        }
        cfg_path_general = tmp_path / "cfg_general.yaml"
        _save_yaml(cfg_path_general, cfg_general)
        proc_dir_general = _run_general(cfg_path_general, tmp_path / "general")

        # legacy path: no dataset_type, --modality flag
        cfg_legacy = {
            "data": data_cfg,
            "processor": processor_cfg,
            "setup": {"output_dir": str(tmp_path / "legacy")},
        }
        cfg_path_legacy = tmp_path / "cfg_legacy.yaml"
        _save_yaml(cfg_path_legacy, cfg_legacy)
        proc_dir_legacy = _run_legacy(legacy_modality, cfg_path_legacy, tmp_path / "legacy")

        _assert_processors_equivalent(
            _load_processor_dict(proc_dir_general),
            _load_processor_dict(proc_dir_legacy),
        )

    def test_processor_json_byte_identical(
        self, dataset_type, legacy_modality, processor_cfg, data_cfg, tmp_path
    ):
        """processor_config.json must be byte-for-byte identical between both paths."""
        cfg_general = {
            "data": {"dataset_type": dataset_type, **data_cfg},
            "processor": processor_cfg,
            "setup": {"output_dir": str(tmp_path / "general")},
        }
        cfg_path_general = tmp_path / "cfg_general.yaml"
        _save_yaml(cfg_path_general, cfg_general)
        proc_dir_general = _run_general(cfg_path_general, tmp_path / "general")

        cfg_legacy = {
            "data": data_cfg,
            "processor": processor_cfg,
            "setup": {"output_dir": str(tmp_path / "legacy")},
        }
        cfg_path_legacy = tmp_path / "cfg_legacy.yaml"
        _save_yaml(cfg_path_legacy, cfg_legacy)
        proc_dir_legacy = _run_legacy(legacy_modality, cfg_path_legacy, tmp_path / "legacy")

        with open(proc_dir_general / "processor_config.json") as f:
            content_general = f.read()
        with open(proc_dir_legacy / "processor_config.json") as f:
            content_legacy = f.read()

        assert content_general == content_legacy, (
            f"[{dataset_type}] processor_config.json differs between general and legacy paths"
        )
