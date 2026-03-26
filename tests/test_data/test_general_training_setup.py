"""Tests for general_training_setup._build_dataset_map and related routing logic."""

import pytest

from multimodalhugs.training_setup.general_training_setup import _build_dataset_map


# ---------------------------------------------------------------------------
# Expected dataset_type values and their corresponding classes
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "pose2text",
    "video2text",
    "features2text",
    "signwriting2text",
    "image2text",
    "text2text",
}


class TestBuildDatasetMapKeys:
    """_build_dataset_map returns all expected dataset_type keys."""

    def test_returns_all_expected_keys(self):
        dataset_map = _build_dataset_map()
        assert set(dataset_map.keys()) == EXPECTED_KEYS

    def test_no_legacy_signwriting_key(self):
        """'signwriting' (without 2text) must not appear — old name was renamed."""
        dataset_map = _build_dataset_map()
        assert "signwriting" not in dataset_map

    def test_no_bilingual_image2text_key(self):
        """'bilingual_image2text' must not appear — renamed to 'image2text'."""
        dataset_map = _build_dataset_map()
        assert "bilingual_image2text" not in dataset_map

    def test_no_bilingual_text2text_key(self):
        """'bilingual_text2text' must not appear — renamed to 'text2text'."""
        dataset_map = _build_dataset_map()
        assert "bilingual_text2text" not in dataset_map


class TestBuildDatasetMapValues:
    """Each entry maps to a (DatasetClass, DataConfigClass) 2-tuple."""

    def test_each_entry_is_two_tuple(self):
        for key, value in _build_dataset_map().items():
            assert isinstance(value, tuple) and len(value) == 2, (
                f"Expected 2-tuple for '{key}', got {value!r}"
            )

    def test_pose2text_classes(self):
        from multimodalhugs.data.datasets.pose2text import Pose2TextDataset, Pose2TextDataConfig
        dataset_cls, config_cls = _build_dataset_map()["pose2text"]
        assert dataset_cls is Pose2TextDataset
        assert config_cls is Pose2TextDataConfig

    def test_video2text_classes(self):
        from multimodalhugs.data.datasets.video2text import Video2TextDataset, Video2TextDataConfig
        dataset_cls, config_cls = _build_dataset_map()["video2text"]
        assert dataset_cls is Video2TextDataset
        assert config_cls is Video2TextDataConfig

    def test_features2text_classes(self):
        from multimodalhugs.data.datasets.features2text import Features2TextDataset, Features2TextDataConfig
        dataset_cls, config_cls = _build_dataset_map()["features2text"]
        assert dataset_cls is Features2TextDataset
        assert config_cls is Features2TextDataConfig

    def test_signwriting2text_classes(self):
        from multimodalhugs.data.datasets.signwriting import SignWritingDataset
        from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import MultimodalDataConfig
        dataset_cls, config_cls = _build_dataset_map()["signwriting2text"]
        assert dataset_cls is SignWritingDataset
        assert config_cls is MultimodalDataConfig

    def test_image2text_classes(self):
        from multimodalhugs.data.datasets.bilingual_image2text import (
            BilingualImage2TextDataset, BilingualImage2textMTDataConfig,
        )
        dataset_cls, config_cls = _build_dataset_map()["image2text"]
        assert dataset_cls is BilingualImage2TextDataset
        assert config_cls is BilingualImage2textMTDataConfig

    def test_text2text_classes(self):
        from multimodalhugs.data.datasets.bilingual_text2text import (
            BilingualText2TextDataset, BilingualText2textMTDataConfig,
        )
        dataset_cls, config_cls = _build_dataset_map()["text2text"]
        assert dataset_cls is BilingualText2TextDataset
        assert config_cls is BilingualText2textMTDataConfig


class TestBuildDatasetMapMatchesModalityMap:
    """dataset_type values must mirror the CLI --modality values exactly."""

    def test_all_modality_map_keys_covered(self):
        """Every --modality value has a corresponding dataset_type."""
        modality_map_keys = {
            "pose2text", "video2text", "features2text",
            "signwriting2text", "image2text", "text2text",
        }
        dataset_map = _build_dataset_map()
        assert modality_map_keys == set(dataset_map.keys())


class TestGeneralTrainingSetupMainValidation:
    """main() raises informative errors for bad configs."""

    def test_missing_dataset_type_raises(self, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"data": {}, "setup": {"output_dir": str(tmp_path)}})
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        from multimodalhugs.training_setup.general_training_setup import main
        with pytest.raises(ValueError, match="data.dataset_type"):
            main(str(cfg_path), do_dataset=True, do_processor=False, do_model=False)

    def test_unknown_dataset_type_raises(self, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "data": {"dataset_type": "nonexistent_modality"},
            "setup": {"output_dir": str(tmp_path)},
        })
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        from multimodalhugs.training_setup.general_training_setup import main
        with pytest.raises(ValueError, match="nonexistent_modality"):
            main(str(cfg_path), do_dataset=True, do_processor=False, do_model=False)

    def test_missing_processor_config_raises(self, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"setup": {"output_dir": str(tmp_path)}})
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        from multimodalhugs.training_setup.general_training_setup import main
        with pytest.raises(ValueError, match="processor"):
            main(str(cfg_path), do_dataset=False, do_processor=True, do_model=False)

    def test_default_dataset_type_fallback(self, tmp_path):
        """When data.dataset_type is absent, default_dataset_type is used instead of raising."""
        from omegaconf import OmegaConf
        # Config has no dataset_type — would raise ValueError without default_dataset_type
        cfg = OmegaConf.create({
            "data": {},
            "setup": {"output_dir": str(tmp_path)},
        })
        cfg_path = tmp_path / "cfg.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        from multimodalhugs.training_setup.general_training_setup import main
        # Without default_dataset_type this raises; with it, it proceeds past the
        # dataset_type check (the dataset step may succeed or fail for other reasons).
        try:
            main(
                str(cfg_path),
                do_dataset=True,
                do_processor=False,
                do_model=False,
                default_dataset_type="pose2text",
            )
        except ValueError as exc:
            assert "data.dataset_type is required" not in str(exc), (
                "default_dataset_type fallback should have prevented the 'required' error"
            )
