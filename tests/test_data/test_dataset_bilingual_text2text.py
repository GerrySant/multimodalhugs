"""Tests for BilingualText2TextDataset."""

import pytest

from multimodalhugs.data.datasets.bilingual_text2text import (
    BilingualText2TextDataset,
    BilingualText2textMTDataConfig,
)


class TestBilingualText2TextDatasetInfo:
    def test_info_returns_correct_features(self):
        config = BilingualText2textMTDataConfig()
        ds = BilingualText2TextDataset(config=config)
        info = ds._info()
        feature_keys = set(info.features.keys())
        assert "signal" in feature_keys
        assert "encoder_prompt" in feature_keys
        assert "decoder_prompt" in feature_keys
        assert "output" in feature_keys


class TestBilingualText2TextSplitGenerators:
    def test_all_metadata_files(self, text2text_tsv):
        config = BilingualText2textMTDataConfig(
            train_metadata_file=text2text_tsv,
            validation_metadata_file=text2text_tsv,
            test_metadata_file=text2text_tsv,
        )
        ds = BilingualText2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 3

    def test_only_train(self, text2text_tsv):
        config = BilingualText2textMTDataConfig(
            train_metadata_file=text2text_tsv,
        )
        ds = BilingualText2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 1

    def test_no_metadata_files(self):
        config = BilingualText2textMTDataConfig()
        ds = BilingualText2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 0


class TestBilingualText2TextGenerateExamples:
    def test_yields_correct_count(self, text2text_tsv):
        config = BilingualText2textMTDataConfig()
        ds = BilingualText2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=text2text_tsv, split="train")
        )
        assert len(examples) == 3

    def test_yields_correct_keys(self, text2text_tsv):
        config = BilingualText2textMTDataConfig()
        ds = BilingualText2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=text2text_tsv, split="train")
        )
        idx, example = examples[0]
        assert "signal" in example
        assert "encoder_prompt" in example
        assert "decoder_prompt" in example
        assert "output" in example

    def test_correct_values(self, text2text_tsv):
        config = BilingualText2textMTDataConfig()
        ds = BilingualText2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=text2text_tsv, split="train")
        )
        _, example = examples[0]
        assert example["signal"] == "Hello world"
        assert example["encoder_prompt"] == "translate:"
        assert example["decoder_prompt"] == "de:"
        assert example["output"] == "Hallo Welt"

    def test_missing_prompts_default_to_empty(self, tmp_path):
        """Test that missing encoder/decoder prompts default to empty string."""
        tsv_path = tmp_path / "no_prompts.tsv"
        with open(tsv_path, "w") as f:
            f.write("signal\toutput\n")
            f.write("Hello\tHallo\n")
        config = BilingualText2textMTDataConfig()
        ds = BilingualText2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=str(tsv_path), split="train")
        )
        _, example = examples[0]
        assert example["encoder_prompt"] == ""
        assert example["decoder_prompt"] == ""


class TestBilingualText2TextMaxSourceTokensBug:
    """Document the bug in bilingual_text2text.py:190.

    duration_filter(self.max_source_tokens, sample) passes args in wrong order.
    The function signature is: duration_filter(sample, min_frames, max_frames).
    So self.max_source_tokens is passed as 'sample', and actual sample as 'min_frames'.
    This means the filter will fail with an error when max_source_tokens is set.
    """

    def test_max_source_tokens_filter_is_buggy(self, text2text_tsv):
        """When max_source_tokens is set, the filter call passes args in wrong order."""
        config = BilingualText2textMTDataConfig(max_source_tokens=100)
        ds = BilingualText2TextDataset(config=config)
        # The bug causes duration_filter to be called as:
        # duration_filter(100, sample) → sample["DURATION"] tries int(100)["DURATION"] → TypeError
        with pytest.raises((TypeError, AttributeError)):
            list(ds._generate_examples(metafile_path=text2text_tsv, split="train"))
