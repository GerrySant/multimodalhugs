"""Tests for multimodalhugs/data/utils.py utility functions."""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf
from PIL import Image
from types import SimpleNamespace

from multimodalhugs.data.utils import (
    string_to_list,
    pad_and_create_mask,
    check_columns,
    contains_empty,
    file_exists_filter,
    duration_filter,
    split_sentence,
    create_image,
    normalize_images,
    get_images,
    gather_appropriate_data_cfg,
    get_all_dataclass_fields,
    build_merged_omegaconf_config,
    resolve_and_update_config,
    center_image_on_white_background,
)
from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import (
    MultimodalDataConfig,
)

from tests.test_data.conftest import FONT_PATH


# ============================================================================
# string_to_list
# ============================================================================
class TestStringToList:
    def test_valid_list_string(self):
        assert string_to_list("[1, 2, 3]") == [1, 2, 3]

    def test_valid_nested_list(self):
        assert string_to_list("[[1, 2], [3, 4]]") == [[1, 2], [3, 4]]

    def test_invalid_string_returns_none(self):
        assert string_to_list("not a list") is None

    def test_empty_list(self):
        assert string_to_list("[]") == []

    def test_float_list(self):
        assert string_to_list("[0.5, 0.5, 0.5]") == [0.5, 0.5, 0.5]


# ============================================================================
# pad_and_create_mask
# ============================================================================
class TestPadAndCreateMask:
    def test_same_length_tensors(self):
        tensors = [torch.ones(5, 3), torch.ones(5, 3)]
        padded, mask = pad_and_create_mask(tensors)
        assert padded.shape == (2, 5, 3)
        assert mask.shape == (2, 5)
        assert mask.sum().item() == 10  # all ones

    def test_different_length_tensors(self):
        tensors = [torch.ones(3, 4), torch.ones(5, 4)]
        padded, mask = pad_and_create_mask(tensors)
        assert padded.shape == (2, 5, 4)
        assert mask.shape == (2, 5)
        # First tensor: 3 real frames, 2 padded
        assert mask[0].tolist() == [1, 1, 1, 0, 0]
        assert mask[1].tolist() == [1, 1, 1, 1, 1]
        # Padding should be zeros
        assert (padded[0, 3:] == 0).all()

    def test_single_tensor(self):
        tensors = [torch.ones(4, 2)]
        padded, mask = pad_and_create_mask(tensors)
        assert padded.shape == (1, 4, 2)
        assert mask.shape == (1, 4)
        assert mask.sum().item() == 4


# ============================================================================
# check_columns
# ============================================================================
class TestCheckColumns:
    def test_dataframe_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert check_columns(df, ["a", "b"]) is True

    def test_dataframe_missing_col(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert check_columns(df, ["a", "c"]) is False

    def test_hf_dataset_all_present(self):
        ds = HFDataset.from_dict({"a": [1], "b": [2]})
        assert check_columns(ds, ["a", "b"]) is True

    def test_hf_dataset_missing_col(self):
        ds = HFDataset.from_dict({"a": [1]})
        assert check_columns(ds, ["a", "b"]) is False


# ============================================================================
# contains_empty
# ============================================================================
class TestContainsEmpty:
    def test_empty_string(self):
        assert contains_empty({"a": "hello", "b": ""}) is True

    def test_none_value(self):
        assert contains_empty({"a": "hello", "b": None}) is True

    def test_all_populated(self):
        assert contains_empty({"a": "hello", "b": "world"}) is False


# ============================================================================
# file_exists_filter
# ============================================================================
class TestFileExistsFilter:
    def test_existing_file(self, tmp_path):
        path = str(tmp_path / "exists.txt")
        with open(path, "w") as f:
            f.write("test")
        assert file_exists_filter("signal", {"signal": path}) is True

    def test_nonexistent_file(self):
        assert (
            file_exists_filter("signal", {"signal": "/nonexistent/path.txt"}) is False
        )


# ============================================================================
# duration_filter
# ============================================================================
class TestDurationFilter:
    def test_no_bounds(self):
        assert duration_filter({"DURATION": 100}) is True

    def test_within_bounds(self):
        assert duration_filter({"DURATION": 50}, min_frames=10, max_frames=100) is True

    def test_below_min(self):
        assert duration_filter({"DURATION": 5}, min_frames=10) is False

    def test_above_max(self):
        assert duration_filter({"DURATION": 200}, max_frames=100) is False

    def test_only_min(self):
        assert duration_filter({"DURATION": 50}, min_frames=10) is True

    def test_only_max(self):
        assert duration_filter({"DURATION": 50}, max_frames=100) is True

    def test_at_exact_min(self):
        assert duration_filter({"DURATION": 10}, min_frames=10) is True

    def test_at_exact_max(self):
        assert duration_filter({"DURATION": 100}, max_frames=100) is True


# ============================================================================
# split_sentence
# ============================================================================
class TestSplitSentence:
    def test_basic_split(self):
        result = split_sentence("Hello world")
        assert result == ["Hello", "world"]

    def test_punctuation_handling(self):
        result = split_sentence("Hello, world!")
        assert "Hello" in result
        assert "world" in result
        # Punctuation should be preserved as tokens
        assert "," in result
        assert "!" in result


# ============================================================================
# create_image
# ============================================================================
class TestCreateImage:
    def test_output_shape(self):
        img = create_image("test", FONT_PATH, img_size=(224, 224))
        assert img.shape == (224, 224, 3)

    def test_custom_size(self):
        img = create_image("A", FONT_PATH, img_size=(64, 64))
        assert img.shape == (64, 64, 3)


# ============================================================================
# normalize_images
# ============================================================================
class TestNormalizeImages:
    def test_normalized_values(self):
        images = np.random.randint(0, 255, (2, 64, 64, 3)).astype(np.float64)
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        result = normalize_images(images, mean, std)
        # Values should be in range approximately [-1, 1]
        assert result.min() >= -1.1
        assert result.max() <= 1.1


# ============================================================================
# get_images
# ============================================================================
class TestGetImages:
    def test_multi_word_string(self):
        result = get_images(
            "Hello world",
            font_path=FONT_PATH,
            width=224,
            height=224,
            normalize_image=False,
            mean=None,
            std=None,
        )
        # Should produce one image per word
        assert result.ndim == 4  # (N_words, C, H, W)
        assert result.shape[0] == 2  # "Hello" and "world"
        assert result.shape[1] == 3  # RGB channels
        assert result.shape[2] == 224
        assert result.shape[3] == 224


# ============================================================================
# gather_appropriate_data_cfg
# ============================================================================
class TestGatherAppropriateDataCfg:
    def test_obj_with_data_attr(self):
        cfg = SimpleNamespace(data={"key": "value"})
        assert gather_appropriate_data_cfg(cfg) == {"key": "value"}

    def test_obj_with_dataset_attr(self):
        cfg = SimpleNamespace(dataset={"key": "value"})
        assert gather_appropriate_data_cfg(cfg) == {"key": "value"}

    def test_plain_obj_fallback(self):
        cfg = SimpleNamespace(other="value")
        result = gather_appropriate_data_cfg(cfg)
        assert hasattr(result, "other")

    def test_none_returns_empty_dict(self):
        assert gather_appropriate_data_cfg(None) == {}

    def test_dict_with_data_key(self):
        cfg = {"data": {"key": "value"}, "other": 1}
        assert gather_appropriate_data_cfg(cfg) == {"key": "value"}


# ============================================================================
# get_all_dataclass_fields
# ============================================================================
class TestGetAllDataclassFields:
    def test_multimodal_data_config_fields(self):
        result = get_all_dataclass_fields(MultimodalDataConfig)
        assert "name" in result
        assert "train_metadata_file" in result
        assert "shuffle" in result

    def test_inherited_fields_from_subclass(self):
        from multimodalhugs.data.datasets.pose2text import Pose2TextDataConfig

        result = get_all_dataclass_fields(Pose2TextDataConfig)
        # Own fields
        assert "max_frames" in result
        assert "min_frames" in result
        # Inherited from MultimodalDataConfig
        assert "train_metadata_file" in result

    def test_non_dataclass_returns_empty(self):
        assert get_all_dataclass_fields(int) == set()


# ============================================================================
# build_merged_omegaconf_config
# ============================================================================
class TestBuildMergedOmegaconfConfig:
    def test_valid_keys_kept(self):
        cfg = OmegaConf.create({"name": "test", "shuffle": False})
        valid, extra, omega = build_merged_omegaconf_config(MultimodalDataConfig, cfg)
        assert "name" in valid
        assert "shuffle" in valid

    def test_extra_keys_separated(self):
        cfg = OmegaConf.create({"name": "test", "unknown_key": 42})
        valid, extra, omega = build_merged_omegaconf_config(MultimodalDataConfig, cfg)
        assert "unknown_key" in extra
        assert "unknown_key" not in valid

    def test_overrides_take_precedence(self):
        cfg = OmegaConf.create({"name": "original"})
        valid, extra, omega = build_merged_omegaconf_config(
            MultimodalDataConfig, cfg, name="overridden"
        )
        assert valid["name"] == "overridden"


# ============================================================================
# resolve_and_update_config
# ============================================================================
class TestResolveAndUpdateConfig:
    def test_config_none_creates_new(self):
        config, remaining = resolve_and_update_config(
            MultimodalDataConfig, config=None, kwargs={}
        )
        assert isinstance(config, MultimodalDataConfig)

    def test_existing_config_gets_updated(self):
        existing = MultimodalDataConfig()
        config, remaining = resolve_and_update_config(
            MultimodalDataConfig, config=existing, kwargs={"shuffle": False}
        )
        assert config.shuffle is False
        assert config is existing  # same object, mutated in place

    def test_non_config_kwargs_returned(self):
        config, remaining = resolve_and_update_config(
            MultimodalDataConfig,
            config=None,
            kwargs={"shuffle": True, "extra_arg": 123},
        )
        assert "extra_arg" in remaining
        assert "shuffle" not in remaining


# ============================================================================
# center_image_on_white_background
# ============================================================================
class TestCenterImageOnWhiteBackground:
    def test_output_size_matches_target(self):
        # Create a small RGBA image (must be RGBA for paste with mask)
        original = Image.new("RGBA", (30, 30), color="blue")
        result = center_image_on_white_background(
            original, target_width=100, target_height=100
        )
        assert result.size == (100, 100)

    def test_different_target_sizes(self):
        original = Image.new("RGBA", (50, 25), color="green")
        result = center_image_on_white_background(
            original, target_width=200, target_height=150
        )
        assert result.size == (200, 150)
