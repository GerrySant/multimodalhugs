"""Tests for BilingualImage2TextDataset."""

from multimodalhugs.data.datasets.bilingual_image2text import (
    BilingualImage2TextDataset,
    BilingualImage2textMTDataConfig,
)


class TestBilingualImage2TextDatasetInfo:
    def test_info_returns_correct_features(self):
        config = BilingualImage2textMTDataConfig()
        ds = BilingualImage2TextDataset(config=config)
        info = ds._info()
        feature_keys = set(info.features.keys())
        assert "signal" in feature_keys
        assert "signal_start" in feature_keys
        assert "signal_end" in feature_keys
        assert "encoder_prompt" in feature_keys
        assert "decoder_prompt" in feature_keys
        assert "output" in feature_keys


class TestBilingualImage2TextSplitGenerators:
    def test_all_splits(self, image2text_tsv):
        config = BilingualImage2textMTDataConfig(
            train_metadata_file=image2text_tsv,
            validation_metadata_file=image2text_tsv,
            test_metadata_file=image2text_tsv,
        )
        ds = BilingualImage2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 3


class TestBilingualImage2TextGenerateExamples:
    def test_yields_correct_count(self, image2text_tsv):
        config = BilingualImage2textMTDataConfig()
        ds = BilingualImage2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=image2text_tsv, split="train")
        )
        assert len(examples) == 2

    def test_yields_correct_fields(self, image2text_tsv):
        config = BilingualImage2textMTDataConfig()
        ds = BilingualImage2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=image2text_tsv, split="train")
        )
        _, example = examples[0]
        assert "signal" in example
        assert "signal_start" in example
        assert "signal_end" in example
        assert "output" in example

    def test_signal_path_preserved(self, image2text_tsv):
        """Signal should contain the image file path as-is."""
        config = BilingualImage2textMTDataConfig()
        ds = BilingualImage2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=image2text_tsv, split="train")
        )
        _, example = examples[0]
        assert example["signal"].endswith(".png")
