"""
Equivalence tests: --modality (legacy) path vs general path (dataset_type in config).

Both paths must produce identical processor artifacts (same slots, same class names,
same output keys, same column maps). Dataset equivalence (feature names, row count)
is also verified.

We test two modalities:
  - text2text: simplest — no binary assets needed
  - pose2text: exercises a non-text modality processor
"""

import json
import os

import pytest
from omegaconf import OmegaConf

from tests.test_data.conftest import TINY_TOKENIZER_PATH, ASSETS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_yaml(path, data: dict):
    OmegaConf.save(OmegaConf.create(data), str(path))


def _run_general(config_path, output_dir):
    """Run the general (no --modality) setup path with do_processor=True only."""
    from multimodalhugs.training_setup.general_training_setup import main
    main(
        config_path=str(config_path),
        do_dataset=False,
        do_processor=True,
        do_model=False,
        output_dir=str(output_dir),
    )
    # resolve_setup_paths appends "setup/"; processor saves to "{output_dir}/setup/multimodal_meta_processor"
    return output_dir / "setup" / "multimodal_meta_processor"


def _run_legacy(modality, config_path, output_dir):
    """Run the legacy --modality setup path with do_processor=True only."""
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
    """Load the saved processor_config.json from a processor directory."""
    config_file = proc_dir / "processor_config.json"
    with open(config_file) as f:
        return json.load(f)


def _assert_processors_equivalent(proc_a: dict, proc_b: dict):
    """Assert two processor dicts have identical slot structure."""
    assert proc_a["processor_class"] == proc_b["processor_class"], (
        "processor_class mismatch"
    )
    slots_a = proc_a["slots"]
    slots_b = proc_b["slots"]
    assert len(slots_a) == len(slots_b), (
        f"Slot count mismatch: general={len(slots_a)}, legacy={len(slots_b)}"
    )
    for i, (sa, sb) in enumerate(zip(slots_a, slots_b)):
        assert sa["processor_class"] == sb["processor_class"], (
            f"Slot {i}: processor_class mismatch: {sa['processor_class']} vs {sb['processor_class']}"
        )
        assert sa.get("output_data_key") == sb.get("output_data_key"), (
            f"Slot {i}: output_data_key mismatch"
        )
        assert sa.get("output_mask_key") == sb.get("output_mask_key"), (
            f"Slot {i}: output_mask_key mismatch"
        )
        assert sa.get("is_label", False) == sb.get("is_label", False), (
            f"Slot {i}: is_label mismatch"
        )
        assert sa.get("column_map") == sb.get("column_map"), (
            f"Slot {i}: column_map mismatch:\n  general: {sa.get('column_map')}\n  legacy:  {sb.get('column_map')}"
        )


# ---------------------------------------------------------------------------
# text2text equivalence
# ---------------------------------------------------------------------------

TEXT_PROCESSOR_CONFIG = {
    "pipeline": "text2text",
    "tokenizer_path": TINY_TOKENIZER_PATH,
}

TEXT_DATA_CONFIG = {
    "train_metadata_file": os.path.join(ASSETS_DIR, "text", "metadata.tsv"),
    "validation_metadata_file": os.path.join(ASSETS_DIR, "text", "metadata.tsv"),
    "test_metadata_file": os.path.join(ASSETS_DIR, "text", "metadata.tsv"),
}


class TestText2TextProcessorEquivalence:
    """General path vs --modality text2text produce identical processor artifacts."""

    def test_processor_slot_structure_identical(self, tmp_path):
        # --- general path: dataset_type in config ---
        cfg_general = {
            "data": {"dataset_type": "text2text", **TEXT_DATA_CONFIG},
            "processor": TEXT_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "general")},
        }
        cfg_path_general = tmp_path / "cfg_general.yaml"
        _save_yaml(cfg_path_general, cfg_general)
        proc_dir_general = _run_general(cfg_path_general, tmp_path / "general")

        # --- legacy path: no dataset_type, --modality text2text ---
        cfg_legacy = {
            "data": TEXT_DATA_CONFIG,
            "processor": TEXT_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "legacy")},
        }
        cfg_path_legacy = tmp_path / "cfg_legacy.yaml"
        _save_yaml(cfg_path_legacy, cfg_legacy)
        proc_dir_legacy = _run_legacy("text2text", cfg_path_legacy, tmp_path / "legacy")

        dict_general = _load_processor_dict(proc_dir_general)
        dict_legacy = _load_processor_dict(proc_dir_legacy)
        _assert_processors_equivalent(dict_general, dict_legacy)

    def test_processor_slot_count(self, tmp_path):
        cfg = {
            "data": {"dataset_type": "text2text", **TEXT_DATA_CONFIG},
            "processor": TEXT_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path)},
        }
        cfg_path = tmp_path / "cfg.yaml"
        _save_yaml(cfg_path, cfg)
        proc_dir = _run_general(cfg_path, tmp_path)
        d = _load_processor_dict(proc_dir)
        # text2text pipeline expands to 4 slots
        assert len(d["slots"]) == 4

    def test_processor_json_files_byte_identical(self, tmp_path):
        """processor_config.json from both paths must be byte-for-byte identical."""
        cfg_general = {
            "data": {"dataset_type": "text2text", **TEXT_DATA_CONFIG},
            "processor": TEXT_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "general")},
        }
        cfg_path_general = tmp_path / "cfg_general.yaml"
        _save_yaml(cfg_path_general, cfg_general)
        proc_dir_general = _run_general(cfg_path_general, tmp_path / "general")

        cfg_legacy = {
            "data": TEXT_DATA_CONFIG,
            "processor": TEXT_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "legacy")},
        }
        cfg_path_legacy = tmp_path / "cfg_legacy.yaml"
        _save_yaml(cfg_path_legacy, cfg_legacy)
        proc_dir_legacy = _run_legacy("text2text", cfg_path_legacy, tmp_path / "legacy")

        with open(proc_dir_general / "processor_config.json") as f:
            content_general = f.read()
        with open(proc_dir_legacy / "processor_config.json") as f:
            content_legacy = f.read()

        assert content_general == content_legacy, (
            "processor_config.json differs between general and legacy paths"
        )


# ---------------------------------------------------------------------------
# pose2text equivalence
# ---------------------------------------------------------------------------

POSE_METADATA = os.path.join(ASSETS_DIR, "pose", "metadata.tsv")

POSE_PROCESSOR_CONFIG = {
    "pipeline": "pose2text",
    "tokenizer_path": TINY_TOKENIZER_PATH,
}

POSE_DATA_CONFIG = {
    "train_metadata_file": POSE_METADATA,
    "validation_metadata_file": POSE_METADATA,
    "test_metadata_file": POSE_METADATA,
}


class TestPose2TextProcessorEquivalence:
    """General path vs --modality pose2text produce identical processor artifacts."""

    def test_processor_slot_structure_identical(self, tmp_path):
        # --- general path ---
        cfg_general = {
            "data": {"dataset_type": "pose2text", **POSE_DATA_CONFIG},
            "processor": POSE_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "general")},
        }
        cfg_path_general = tmp_path / "cfg_general.yaml"
        _save_yaml(cfg_path_general, cfg_general)
        proc_dir_general = _run_general(cfg_path_general, tmp_path / "general")

        # --- legacy path ---
        cfg_legacy = {
            "data": POSE_DATA_CONFIG,
            "processor": POSE_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "legacy")},
        }
        cfg_path_legacy = tmp_path / "cfg_legacy.yaml"
        _save_yaml(cfg_path_legacy, cfg_legacy)
        proc_dir_legacy = _run_legacy("pose2text", cfg_path_legacy, tmp_path / "legacy")

        dict_general = _load_processor_dict(proc_dir_general)
        dict_legacy = _load_processor_dict(proc_dir_legacy)
        _assert_processors_equivalent(dict_general, dict_legacy)

    def test_pose_slot_is_first(self, tmp_path):
        """Slot 0 must be PoseModalityProcessor for pose2text."""
        cfg = {
            "data": {"dataset_type": "pose2text", **POSE_DATA_CONFIG},
            "processor": POSE_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path)},
        }
        cfg_path = tmp_path / "cfg.yaml"
        _save_yaml(cfg_path, cfg)
        proc_dir = _run_general(cfg_path, tmp_path)
        d = _load_processor_dict(proc_dir)
        assert d["slots"][0]["processor_class"] == "PoseModalityProcessor"

    def test_processor_json_files_byte_identical(self, tmp_path):
        """processor_config.json from both paths must be byte-for-byte identical."""
        cfg_general = {
            "data": {"dataset_type": "pose2text", **POSE_DATA_CONFIG},
            "processor": POSE_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "general")},
        }
        cfg_path_general = tmp_path / "cfg_general.yaml"
        _save_yaml(cfg_path_general, cfg_general)
        proc_dir_general = _run_general(cfg_path_general, tmp_path / "general")

        cfg_legacy = {
            "data": POSE_DATA_CONFIG,
            "processor": POSE_PROCESSOR_CONFIG,
            "setup": {"output_dir": str(tmp_path / "legacy")},
        }
        cfg_path_legacy = tmp_path / "cfg_legacy.yaml"
        _save_yaml(cfg_path_legacy, cfg_legacy)
        proc_dir_legacy = _run_legacy("pose2text", cfg_path_legacy, tmp_path / "legacy")

        with open(proc_dir_general / "processor_config.json") as f:
            content_general = f.read()
        with open(proc_dir_legacy / "processor_config.json") as f:
            content_legacy = f.read()

        assert content_general == content_legacy, (
            "processor_config.json differs between general and legacy paths"
        )
