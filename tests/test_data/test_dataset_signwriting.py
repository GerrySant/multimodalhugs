"""Tests for SignWritingDataset."""

from omegaconf import OmegaConf

from multimodalhugs.data.datasets.signwriting import SignWritingDataset
from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import (
    MultimodalDataConfig,
)


class TestSignWritingDatasetInfo:
    def test_info_returns_correct_features(self):
        config = MultimodalDataConfig()
        ds = SignWritingDataset(config=config)
        info = ds._info()
        feature_keys = set(info.features.keys())
        assert "signal" in feature_keys
        assert "signal_start" in feature_keys
        assert "signal_end" in feature_keys
        assert "encoder_prompt" in feature_keys
        assert "decoder_prompt" in feature_keys
        assert "output" in feature_keys


class TestSignWritingSplitGenerators:
    def test_all_splits(self, signwriting_tsv):
        omega = OmegaConf.create(
            {
                "train_metadata_file": signwriting_tsv,
                "validation_metadata_file": signwriting_tsv,
                "test_metadata_file": signwriting_tsv,
            }
        )
        config = MultimodalDataConfig(cfg=omega)
        ds = SignWritingDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 3


class TestSignWritingGenerateExamples:
    def test_yields_correct_count(self, signwriting_tsv):
        config = MultimodalDataConfig()
        ds = SignWritingDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=signwriting_tsv, split="train")
        )
        assert len(examples) == 3

    def test_yields_correct_fields(self, signwriting_tsv):
        config = MultimodalDataConfig()
        ds = SignWritingDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=signwriting_tsv, split="train")
        )
        _, example = examples[0]
        assert "signal" in example
        assert "signal_start" in example
        assert "signal_end" in example
        assert "output" in example

    def test_signal_is_fsw_string(self, signwriting_tsv):
        """Signal should be the FSW string from the TSV, not a file path."""
        config = MultimodalDataConfig()
        ds = SignWritingDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=signwriting_tsv, split="train")
        )
        _, example = examples[0]
        # FSW strings start with 'M'
        assert example["signal"].startswith("M")

    def test_no_file_filtering(self, signwriting_tsv):
        """All rows should be yielded since there's no file I/O filtering."""
        config = MultimodalDataConfig()
        ds = SignWritingDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=signwriting_tsv, split="train")
        )
        assert len(examples) == 3
