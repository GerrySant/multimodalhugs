"""Tests for Video2TextDataset."""

from multimodalhugs.data.datasets.video2text import (
    Video2TextDataset,
    Video2TextDataConfig,
)


class TestVideo2TextDatasetInfo:
    def test_info_returns_correct_features(self):
        config = Video2TextDataConfig()
        ds = Video2TextDataset(config=config)
        info = ds._info()
        feature_keys = set(info.features.keys())
        assert "signal" in feature_keys
        assert "signal_start" in feature_keys
        assert "signal_end" in feature_keys
        assert "encoder_prompt" in feature_keys
        assert "decoder_prompt" in feature_keys
        assert "output" in feature_keys


class TestVideo2TextSplitGenerators:
    def test_all_splits(self, video2text_tsv):
        config = Video2TextDataConfig(
            train_metadata_file=video2text_tsv,
            validation_metadata_file=video2text_tsv,
            test_metadata_file=video2text_tsv,
        )
        ds = Video2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 3

    def test_no_metadata(self):
        config = Video2TextDataConfig()
        ds = Video2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 0


class TestVideo2TextGenerateExamples:
    def test_yields_correct_count(self, video2text_tsv):
        config = Video2TextDataConfig()
        ds = Video2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=video2text_tsv, split="train")
        )
        assert len(examples) == 2

    def test_yields_correct_fields(self, video2text_tsv):
        config = Video2TextDataConfig()
        ds = Video2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=video2text_tsv, split="train")
        )
        _, example = examples[0]
        assert "signal" in example
        assert "signal_start" in example
        assert "signal_end" in example
        assert "output" in example

    def test_file_exists_filter(self, tmp_path, dummy_video_file):
        """Rows with nonexistent signal paths are filtered out."""
        tsv_path = tmp_path / "mixed.tsv"
        with open(tsv_path, "w") as f:
            f.write(
                "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput\n"
            )
            f.write(f"{dummy_video_file}\t0\t0\ttranslate:\tde:\tHello\n")
            f.write("/nonexistent/video.mp4\t0\t0\ttranslate:\tde:\tWorld\n")
        config = Video2TextDataConfig()
        ds = Video2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=str(tsv_path), split="train")
        )
        assert len(examples) == 1

    def test_frame_count_works(self, video2text_tsv):
        """Frame counting via av works on the dummy video."""
        config = Video2TextDataConfig()
        ds = Video2TextDataset(config=config)
        examples = list(
            ds._generate_examples(metafile_path=video2text_tsv, split="train")
        )
        assert len(examples) == 2
        # Each example should have the video path preserved
        _, example = examples[0]
        assert example["signal"].endswith(".mp4")
