"""Tests for merge_config_and_command_args in tasks/translation/utils.py."""

import pytest
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from unittest.mock import patch

from multimodalhugs.tasks.translation.config_classes import ExtendedSeq2SeqTrainingArguments
from multimodalhugs.tasks.translation.utils import merge_config_and_command_args


def _default_training_args():
    parser = HfArgumentParser((ExtendedSeq2SeqTrainingArguments,))
    return parser.parse_dict({"output_dir": "/tmp/test_output"})[0]


class TestMergeConfigDerivedAttributes:
    """Derived TrainingArguments attributes (mixed_precision, etc.) must be recomputed after merge."""

    def test_fp16_from_yaml_sets_mixed_precision(self, tmp_path):
        """fp16: true in YAML must produce mixed_precision='fp16' on the returned args.

        Regression test for the transformers 5.x bug where merge_config_and_command_args
        returned the original CLI-parsed _args instance (with stale mixed_precision='no')
        instead of the freshly constructed extra_args (where __post_init__ set
        mixed_precision='fp16' from fp16=True).
        """
        cfg = OmegaConf.create({"training": {"fp16": True}})
        cfg_path = tmp_path / "config.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        _args = _default_training_args()
        assert _args.fp16 is False
        assert _args.mixed_precision == "no"

        result = merge_config_and_command_args(
            str(cfg_path), ExtendedSeq2SeqTrainingArguments, "training", _args, []
        )

        assert result.fp16 is True
        assert result.mixed_precision == "fp16", (
            "mixed_precision must be 'fp16' when fp16=True is set via YAML; "
            "a stale value of 'no' means __post_init__ did not run on the returned instance"
        )

    def test_bf16_from_yaml_sets_mixed_precision(self, tmp_path):
        cfg = OmegaConf.create({"training": {"bf16": True}})
        cfg_path = tmp_path / "config.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        # _validate_args rejects bf16=True on CPU-only hardware. Patch it out on
        # the exact class being instantiated so the mock is found first in the MRO,
        # regardless of any _validate_args overrides in the transformers hierarchy.
        # mixed_precision is computed by __post_init__ before _validate_args runs,
        # so the patch suppresses only the hardware-check exception.
        with patch.object(ExtendedSeq2SeqTrainingArguments, "_validate_args"):
            result = merge_config_and_command_args(
                str(cfg_path), ExtendedSeq2SeqTrainingArguments, "training", _default_training_args(), []
            )

        assert result.bf16 is True
        assert result.mixed_precision == "bf16"

    def test_no_fp16_in_yaml_keeps_mixed_precision_no(self, tmp_path):
        cfg = OmegaConf.create({"training": {"num_train_epochs": 3}})
        cfg_path = tmp_path / "config.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        result = merge_config_and_command_args(
            str(cfg_path), ExtendedSeq2SeqTrainingArguments, "training", _default_training_args(), []
        )

        assert result.mixed_precision == "no"

    def test_cli_arg_takes_precedence_over_yaml(self, tmp_path):
        """An explicitly CLI-provided field must win over the YAML value."""
        cfg = OmegaConf.create({"training": {"num_train_epochs": 10}})
        cfg_path = tmp_path / "config.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        _args = _default_training_args()
        # Simulate --num_train_epochs 3 on the command line
        _args.num_train_epochs = 3
        remaining_args = ["--num_train_epochs", "3"]

        result = merge_config_and_command_args(
            str(cfg_path), ExtendedSeq2SeqTrainingArguments, "training", _args, remaining_args
        )

        # CLI value (3) must win over YAML value (10)
        assert result.num_train_epochs == 3

    def test_missing_section_returns_args_unchanged(self, tmp_path):
        cfg = OmegaConf.create({"model": {"backbone_type": "m2m_100"}})
        cfg_path = tmp_path / "config.yaml"
        OmegaConf.save(cfg, str(cfg_path))

        _args = _default_training_args()
        result = merge_config_and_command_args(
            str(cfg_path), ExtendedSeq2SeqTrainingArguments, "training", _args, []
        )

        assert result is _args
