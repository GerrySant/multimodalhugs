"""Tests for dataset configuration dataclasses."""

from omegaconf import OmegaConf

from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import (
    MultimodalDataConfig,
)
from multimodalhugs.data.datasets.bilingual_text2text import (
    BilingualText2textMTDataConfig,
)
from multimodalhugs.data.datasets.pose2text import Pose2TextDataConfig
from multimodalhugs.data.datasets.video2text import Video2TextDataConfig
from multimodalhugs.data.datasets.features2text import Features2TextDataConfig
from multimodalhugs.data.datasets.bilingual_image2text import (
    BilingualImage2textMTDataConfig,
)


class TestMultimodalDataConfig:
    def test_defaults(self):
        cfg = MultimodalDataConfig()
        # name defaults to 'default' (from BuilderConfig parent)
        assert cfg.name == "default"
        assert cfg.train_metadata_file is None
        assert cfg.validation_metadata_file is None
        assert cfg.test_metadata_file is None
        assert cfg.shuffle is True

    def test_custom_values_via_omegaconf(self):
        omega = OmegaConf.create({"name": "custom", "shuffle": False})
        cfg = MultimodalDataConfig(cfg=omega)
        assert cfg.name == "custom"
        assert cfg.shuffle is False

    def test_from_omegaconf(self):
        omega = OmegaConf.create(
            {
                "name": "from_omega",
                "train_metadata_file": "/path/to/train.tsv",
                "shuffle": False,
            }
        )
        cfg = MultimodalDataConfig(cfg=omega)
        assert cfg.name == "from_omega"
        assert cfg.train_metadata_file == "/path/to/train.tsv"
        assert cfg.shuffle is False


class TestBilingualText2textMTDataConfig:
    def test_defaults(self):
        cfg = BilingualText2textMTDataConfig()
        assert cfg.name == "default"
        assert cfg.max_source_tokens is None

    def test_custom_max_source_tokens(self):
        cfg = BilingualText2textMTDataConfig(max_source_tokens=100)
        assert cfg.max_source_tokens == 100

    def test_name_from_omegaconf(self):
        omega = OmegaConf.create({"name": "BilingualText2textMTDataConfig"})
        cfg = BilingualText2textMTDataConfig(cfg=omega)
        assert cfg.name == "BilingualText2textMTDataConfig"


class TestPose2TextDataConfig:
    def test_defaults(self):
        cfg = Pose2TextDataConfig()
        assert cfg.name == "default"
        assert cfg.max_frames is None
        assert cfg.min_frames is None

    def test_custom_frame_limits(self):
        cfg = Pose2TextDataConfig(max_frames=500, min_frames=10)
        assert cfg.max_frames == 500
        assert cfg.min_frames == 10


class TestVideo2TextDataConfig:
    def test_defaults(self):
        cfg = Video2TextDataConfig()
        assert cfg.name == "default"
        assert cfg.max_frames is None
        assert cfg.min_frames is None

    def test_custom_frame_limits(self):
        cfg = Video2TextDataConfig(max_frames=300, min_frames=5)
        assert cfg.max_frames == 300
        assert cfg.min_frames == 5


class TestFeatures2TextDataConfig:
    def test_defaults(self):
        cfg = Features2TextDataConfig()
        assert cfg.name == "default"
        assert cfg.max_frames is None
        assert cfg.min_frames is None
        assert cfg.preload_features is False

    def test_preload_features(self):
        cfg = Features2TextDataConfig(preload_features=True)
        assert cfg.preload_features is True


class TestBilingualImage2textMTDataConfig:
    def test_defaults(self):
        cfg = BilingualImage2textMTDataConfig()
        assert cfg.name == "default"

    def test_inherited_defaults(self):
        cfg = BilingualImage2textMTDataConfig()
        assert cfg.train_metadata_file is None
        assert cfg.shuffle is True


class TestConfigFromOmegaConf:
    def test_pose_config_from_omega(self):
        omega = OmegaConf.create({"max_frames": 200, "min_frames": 5})
        cfg = Pose2TextDataConfig(cfg=omega)
        assert cfg.max_frames == 200
        assert cfg.min_frames == 5

    def test_kwargs_override_cfg(self):
        omega = OmegaConf.create({"max_frames": 200})
        cfg = Pose2TextDataConfig(cfg=omega, max_frames=300)
        assert cfg.max_frames == 300
