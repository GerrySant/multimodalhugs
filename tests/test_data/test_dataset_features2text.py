"""Tests for Features2TextDataset."""

from multimodalhugs.data.datasets.features2text import (
    Features2TextDataset,
    Features2TextDataConfig,
)


class TestFeatures2TextDatasetInfo:
    def test_info_returns_correct_features(self):
        config = Features2TextDataConfig()
        ds = Features2TextDataset(config=config)
        info = ds._info()
        feature_keys = set(info.features.keys())
        assert "signal" in feature_keys
        assert "signal_start" in feature_keys
        assert "signal_end" in feature_keys
        assert "encoder_prompt" in feature_keys
        assert "decoder_prompt" in feature_keys
        assert "output" in feature_keys


class TestFeatures2TextSplitGenerators:
    def test_all_splits(self, features2text_tsv):
        config = Features2TextDataConfig(
            train_metadata_file=features2text_tsv,
            validation_metadata_file=features2text_tsv,
            test_metadata_file=features2text_tsv,
        )
        ds = Features2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 3


class TestFeatures2TextGenerateExamples:
    def test_yields_correct_count(self, features2text_tsv):
        config = Features2TextDataConfig()
        ds = Features2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=features2text_tsv, split="train")
        )
        assert len(examples) == 3

    def test_yields_correct_fields(self, features2text_tsv):
        config = Features2TextDataConfig()
        ds = Features2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=features2text_tsv, split="train")
        )
        _, example = examples[0]
        assert "signal" in example
        assert "signal_start" in example
        assert "signal_end" in example
        assert "output" in example

    def test_file_exists_filter(self, tmp_path, dummy_npy_file):
        """Rows with nonexistent signal paths are filtered out."""
        tsv_path = tmp_path / "mixed.tsv"
        with open(tsv_path, "w") as f:
            f.write(
                "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput\n"
            )
            f.write(f"{dummy_npy_file}\t0\t0\ttranslate:\tde:\tHello\n")
            f.write("/nonexistent/features.npy\t0\t0\ttranslate:\tde:\tWorld\n")
        config = Features2TextDataConfig()
        ds = Features2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=str(tsv_path), split="train")
        )
        assert len(examples) == 1

    def test_max_frames_filtering(self, features2text_tsv):
        """max_frames filtering based on array shape[0]."""
        # Dummy npy has shape (10, 64), so max_frames=5 should filter all out
        config = Features2TextDataConfig(max_frames=5)
        ds = Features2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=features2text_tsv, split="train")
        )
        assert len(examples) == 0

    def test_max_frames_keeps_within_range(self, features2text_tsv):
        """max_frames filtering keeps samples within range."""
        config = Features2TextDataConfig(max_frames=100)
        ds = Features2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=features2text_tsv, split="train")
        )
        assert len(examples) == 3

    def test_preload_features_false_keeps_path(self, features2text_tsv):
        """With preload_features=False, signal remains a file path."""
        config = Features2TextDataConfig(preload_features=False)
        ds = Features2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=features2text_tsv, split="train")
        )
        _, example = examples[0]
        assert isinstance(example["signal"], str)
        assert example["signal"].endswith(".npy")
