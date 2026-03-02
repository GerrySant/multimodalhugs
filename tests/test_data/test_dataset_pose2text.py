"""Tests for Pose2TextDataset."""

from multimodalhugs.data.datasets.pose2text import Pose2TextDataset, Pose2TextDataConfig


class TestPose2TextDatasetInfo:
    def test_info_returns_correct_features(self):
        config = Pose2TextDataConfig()
        ds = Pose2TextDataset(config=config)
        info = ds._info()
        feature_keys = set(info.features.keys())
        assert "signal" in feature_keys
        assert "signal_start" in feature_keys
        assert "signal_end" in feature_keys
        assert "encoder_prompt" in feature_keys
        assert "decoder_prompt" in feature_keys
        assert "output" in feature_keys


class TestPose2TextSplitGenerators:
    def test_all_splits(self, pose2text_tsv):
        config = Pose2TextDataConfig(
            train_metadata_file=pose2text_tsv,
            validation_metadata_file=pose2text_tsv,
            test_metadata_file=pose2text_tsv,
        )
        ds = Pose2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 3

    def test_no_metadata(self):
        config = Pose2TextDataConfig()
        ds = Pose2TextDataset(config=config)
        splits = ds._split_generators(dl_manager=None)
        assert len(splits) == 0


class TestPose2TextGenerateExamples:
    def test_yields_correct_count(self, pose2text_tsv):
        config = Pose2TextDataConfig()
        ds = Pose2TextDataset(config=config)
        examples = list(
            ds._generate_examples(split="train", metafile_path=pose2text_tsv)
        )
        assert len(examples) == 3

    def test_yields_correct_fields(self, pose2text_tsv):
        config = Pose2TextDataConfig()
        ds = Pose2TextDataset(config=config)
        examples = list(
            ds._generate_examples(split="train", metafile_path=pose2text_tsv)
        )
        _, example = examples[0]
        assert "signal" in example
        assert "signal_start" in example
        assert "signal_end" in example
        assert "output" in example

    def test_file_exists_filter_removes_missing(self, tmp_path, dummy_pose_file):
        """Rows with nonexistent signal paths are filtered out."""
        tsv_path = tmp_path / "mixed.tsv"
        with open(tsv_path, "w") as f:
            f.write(
                "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput\n"
            )
            f.write(f"{dummy_pose_file}\t0\t0\ttranslate:\tde:\tHello\n")
            f.write("/nonexistent/path.pose\t0\t0\ttranslate:\tde:\tWorld\n")
        config = Pose2TextDataConfig()
        ds = Pose2TextDataset(config=config)
        examples = list(
            ds._generate_examples(split="train", metafile_path=str(tsv_path))
        )
        assert len(examples) == 1

    def test_max_frames_filtering_train(self, pose2text_tsv):
        """max_frames filtering is applied for train split."""
        # The dummy pose has 10 frames; set max_frames=5 to filter it out
        config = Pose2TextDataConfig(max_frames=5)
        ds = Pose2TextDataset(config=config)
        examples = list(
            ds._generate_examples(split="train", metafile_path=pose2text_tsv)
        )
        assert len(examples) == 0

    def test_max_frames_no_filtering_if_large_enough(self, pose2text_tsv):
        """max_frames filtering keeps samples within range."""
        config = Pose2TextDataConfig(max_frames=100)
        ds = Pose2TextDataset(config=config)
        examples = list(
            ds._generate_examples(split="train", metafile_path=pose2text_tsv)
        )
        assert len(examples) == 3

    def test_test_split_skips_duration_filtering(self, pose2text_tsv):
        """Test split should skip duration filtering (line 228 of source)."""
        config = Pose2TextDataConfig(max_frames=5)
        ds = Pose2TextDataset(config=config)
        examples = list(
            ds._generate_examples(split="test", metafile_path=pose2text_tsv)
        )
        # All 3 examples should remain despite max_frames=5
        assert len(examples) == 3
