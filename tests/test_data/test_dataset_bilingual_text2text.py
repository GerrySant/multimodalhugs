"""Tests for BilingualText2TextDataset."""

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


class TestBilingualText2TextMaxSourceTokensFilter:
    def test_max_source_tokens_filters_long_samples(self, text2text_tsv):
        """Samples with more words than max_source_tokens are filtered out."""
        # "Hello world" has 2 words, "Good morning" has 2, "Thank you" has 2
        # Setting max_source_tokens=1 should filter all of them out
        config = BilingualText2textMTDataConfig(max_source_tokens=1)
        ds = BilingualText2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=text2text_tsv, split="train")
        )
        assert len(examples) == 0

    def test_max_source_tokens_keeps_short_samples(self, text2text_tsv):
        """Samples within max_source_tokens limit are kept."""
        # All samples have 2 words, so max_source_tokens=5 keeps all
        config = BilingualText2textMTDataConfig(max_source_tokens=5)
        ds = BilingualText2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=text2text_tsv, split="train")
        )
        assert len(examples) == 3
