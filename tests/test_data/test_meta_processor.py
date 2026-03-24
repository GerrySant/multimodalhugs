"""
Tests for ProcessorSlot and MultimodalMetaProcessor.

These tests cover the meta-processor layer that composes individual
ModalityProcessors into a full pipeline via a flat list of ProcessorSlots.

Design decisions reflected in these tests
------------------------------------------
* ProcessorSlot binds a ModalityProcessor to:
    - column_map       : which TSV / sample-dict keys to read from
    - output_data_key  : forward() argument name for the data tensor
    - output_mask_key  : forward() argument name for the mask (optional)
    - is_label         : marks this slot as producing a loss target

* MultimodalMetaProcessor is constructed from a flat list of ProcessorSlots:
    - slots : List[ProcessorSlot]

* Backward-compatibility contract
    A MetaProcessor configured for pose→text must produce the same set of
    output keys that Pose2TextTranslationProcessor + DataCollator produced:
        input_frames, attention_mask,
        encoder_prompt, encoder_prompt_length_padding_mask,
        decoder_input_ids, decoder_attention_mask,
        labels

* Labels are produced by the MetaProcessor (via a slot with is_label=True),
    not the DataCollator. The DataCollator only adds decoder_input_ids from
    labels when the model provides prepare_decoder_input_ids_from_labels().
"""

import json
import tempfile

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.meta_processor import (
    ProcessorSlot,
    MultimodalMetaProcessor,
)
from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from multimodalhugs.processors.features_modality_processor import FeaturesModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor
from multimodalhugs.data.datacollators.multimodal_datacollator import (
    DataCollatorMultimodalSeq2Seq,
)


# ---------------------------------------------------------------------------
# Local fixtures (not in conftest — task-specific shapes)
# ---------------------------------------------------------------------------

@pytest.fixture
def text_batch_samples_no_signal():
    """Batch samples for text→text where the encoder input is in 'signal'."""
    return [
        {
            "signal": "Hello world",
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hallo Welt",
        },
        {
            "signal": "Good morning",
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Guten Morgen",
        },
    ]


@pytest.fixture
def multi_input_batch_samples(dummy_pose_file, dummy_video_file):
    """Batch samples for a video+pose→text scenario with two encoder columns."""
    return [
        {
            "video_signal": dummy_video_file,
            "pose_signal": dummy_pose_file,
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "Hello",
        },
        {
            "video_signal": dummy_video_file,
            "pose_signal": dummy_pose_file,
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "translate:",
            "decoder_prompt": "de:",
            "output": "World",
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pose2text_meta(tokenizer):
    """Return a MetaProcessor equivalent to Pose2TextTranslationProcessor."""
    return MultimodalMetaProcessor(
        slots=[
            ProcessorSlot(
                processor=PoseModalityProcessor(reduce_holistic_poses=True),
                output_data_key="input_frames",
                output_mask_key="attention_mask",
                column_map={
                    "signal": "signal",
                    "signal_start": "signal_start",
                    "signal_end": "signal_end",
                },
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
                output_data_key="labels",
                is_label=True,
                column_map={"decoder_prompt": "target_prefix", "output": "target"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
        ],
        tokenizer=tokenizer,
    )


def make_text2text_meta(tokenizer):
    """Return a MetaProcessor equivalent to Text2TextTranslationProcessor."""
    return MultimodalMetaProcessor(
        slots=[
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="input_ids",
                output_mask_key="attention_mask",
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
                output_data_key="labels",
                is_label=True,
                column_map={"decoder_prompt": "target_prefix", "output": "target"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
        ],
        tokenizer=tokenizer,
    )


def _make_features2text_meta(tokenizer, use_cache=False):
    """Build a minimal features-to-text MetaProcessor with 4 slots."""
    return MultimodalMetaProcessor(
        slots=[
            ProcessorSlot(
                processor=FeaturesModalityProcessor(use_cache=use_cache),
                output_data_key="input_frames",
                output_mask_key="attention_mask",
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
                output_data_key="labels",
                is_label=True,
                column_map={"decoder_prompt": "target_prefix", "output": "target"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
        ],
        tokenizer=tokenizer,
    )


def _make_multi_input_meta(tokenizer):
    """Build a MetaProcessor with two encoder slots (e.g. features + text)."""
    return MultimodalMetaProcessor(
        slots=[
            ProcessorSlot(
                processor=FeaturesModalityProcessor(use_cache=False),
                output_data_key="input_frames",
                output_mask_key="attention_mask",
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="secondary_input",
                output_mask_key="secondary_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
                output_data_key="labels",
                is_label=True,
                column_map={"decoder_prompt": "target_prefix", "output": "target"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
        ],
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# ProcessorSlot
# ---------------------------------------------------------------------------

class TestProcessorSlot:

    def test_instantiation_with_required_fields(self):
        slot = ProcessorSlot(
            processor=PoseModalityProcessor(),
            output_data_key="input_frames",
        )
        assert slot.primary_field == "signal"  # default column_map
        assert slot.output_data_key == "input_frames"

    def test_output_mask_key_defaults_to_none(self, tokenizer):
        slot = ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        )
        assert slot.output_mask_key is None

    def test_output_mask_key_can_be_set(self):
        slot = ProcessorSlot(
            processor=PoseModalityProcessor(),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
        )
        assert slot.output_mask_key == "attention_mask"

    def test_primary_field_custom_column_map(self, tokenizer):
        slot = ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        )
        assert slot.primary_field == "decoder_prompt"

    def test_is_label_default_false(self, tokenizer):
        slot = ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
            output_data_key="input_ids",
        )
        assert slot.is_label is False

    def test_is_label_true(self, tokenizer):
        slot = ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        )
        assert slot.is_label is True


# ---------------------------------------------------------------------------
# MultimodalMetaProcessor — pose→text  (backward-compatibility)
# ---------------------------------------------------------------------------

class TestMultimodalMetaProcessorPose2Text:

    def test_call_returns_batch_feature(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_call_produces_input_frames(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        assert "input_frames" in result

    def test_call_produces_attention_mask(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        assert "attention_mask" in result

    def test_call_produces_encoder_prompt(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        assert "encoder_prompt" in result
        assert "encoder_prompt_length_padding_mask" in result

    def test_call_produces_decoder_input_ids(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        assert "decoder_input_ids" in result
        assert "decoder_attention_mask" in result

    def test_call_produces_labels(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        assert "labels" in result

    def test_input_frames_shape(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        batch_size = len(pose_batch_samples)
        assert result["input_frames"].ndim == 3
        assert result["input_frames"].shape[0] == batch_size

    def test_attention_mask_shape(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        batch_size = len(pose_batch_samples)
        assert result["attention_mask"].ndim == 2
        assert result["attention_mask"].shape[0] == batch_size

    def test_labels_shape(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        batch_size = len(pose_batch_samples)
        assert result["labels"].ndim == 2
        assert result["labels"].shape[0] == batch_size

    def test_all_batch_dims_consistent(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        batch_size = len(pose_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )

    def test_transform_get_items_output_converts_path_to_tensor(
        self, tokenizer, dummy_pose_file
    ):
        meta = make_pose2text_meta(tokenizer)
        batch = {
            "signal":         [dummy_pose_file],
            "signal_start":   [0],
            "signal_end":     [0],
            "encoder_prompt": ["translate:"],
            "decoder_prompt": ["de:"],
            "output":         ["Hello"],
        }
        result = meta._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)

    def test_transform_get_items_output_leaves_other_columns_intact(
        self, tokenizer, dummy_pose_file
    ):
        meta = make_pose2text_meta(tokenizer)
        batch = {
            "signal":         [dummy_pose_file],
            "signal_start":   [0],
            "signal_end":     [0],
            "encoder_prompt": ["translate:"],
            "output":         ["Hello"],
        }
        result = meta._transform_get_items_output(batch)
        assert result["encoder_prompt"] == ["translate:"]
        assert result["output"] == ["Hello"]

    def test_label_slot_is_marked(self, tokenizer):
        meta = make_pose2text_meta(tokenizer)
        label_slots = [s for s in meta.slots if s.is_label]
        assert len(label_slots) == 1
        assert label_slots[0].output_data_key == "labels"


# ---------------------------------------------------------------------------
# MultimodalMetaProcessor — text→text  (backward-compatibility)
# ---------------------------------------------------------------------------

class TestMultimodalMetaProcessorText2Text:

    def test_call_returns_batch_feature(self, tokenizer, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        result = meta(text_batch_samples_no_signal)
        assert isinstance(result, BatchFeature)

    def test_call_produces_input_ids(self, tokenizer, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        result = meta(text_batch_samples_no_signal)
        assert "input_ids" in result

    def test_call_produces_attention_mask(self, tokenizer, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        result = meta(text_batch_samples_no_signal)
        assert "attention_mask" in result

    def test_call_produces_labels(self, tokenizer, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        result = meta(text_batch_samples_no_signal)
        assert "labels" in result

    def test_all_batch_dims_consistent(self, tokenizer, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        result = meta(text_batch_samples_no_signal)
        batch_size = len(text_batch_samples_no_signal)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )


# ---------------------------------------------------------------------------
# MultimodalMetaProcessor — multi-input  (new scenario: video+pose→text)
# ---------------------------------------------------------------------------

class TestMultimodalMetaProcessorMultiInput:

    def _make_video_pose_meta(self, tokenizer):
        return MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=VideoModalityProcessor(),
                    output_data_key="video_frames",
                    output_mask_key="video_attention_mask",
                    column_map={
                        "video_signal": "signal",
                        "signal_start": "signal_start",
                        "signal_end": "signal_end",
                    },
                ),
                ProcessorSlot(
                    processor=PoseModalityProcessor(reduce_holistic_poses=True),
                    output_data_key="pose_frames",
                    output_mask_key="pose_attention_mask",
                    column_map={
                        "pose_signal": "signal",
                        "signal_start": "signal_start",
                        "signal_end": "signal_end",
                    },
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
                    output_data_key="labels",
                    is_label=True,
                    column_map={"decoder_prompt": "target_prefix", "output": "target"},
                ),
            ],
            tokenizer=tokenizer,
        )

    def test_call_produces_video_frames(self, tokenizer, multi_input_batch_samples):
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        assert "video_frames" in result

    def test_call_produces_video_mask(self, tokenizer, multi_input_batch_samples):
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        assert "video_attention_mask" in result

    def test_call_produces_pose_frames(self, tokenizer, multi_input_batch_samples):
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        assert "pose_frames" in result

    def test_call_produces_pose_mask(self, tokenizer, multi_input_batch_samples):
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        assert "pose_attention_mask" in result

    def test_call_produces_labels(self, tokenizer, multi_input_batch_samples):
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        assert "labels" in result

    def test_output_keys_match_slot_declarations(
        self, tokenizer, multi_input_batch_samples
    ):
        """Every output_data_key and output_mask_key declared in slots must appear in output."""
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        for slot in meta.slots:
            assert slot.output_data_key in result
            if slot.output_mask_key:
                assert slot.output_mask_key in result

    def test_video_and_pose_masks_are_independent(
        self, tokenizer, multi_input_batch_samples
    ):
        """The two encoder streams must each have their own mask."""
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        assert result["video_attention_mask"] is not result["pose_attention_mask"]

    def test_all_batch_dims_consistent(self, tokenizer, multi_input_batch_samples):
        meta = self._make_video_pose_meta(tokenizer)
        result = meta(multi_input_batch_samples)
        batch_size = len(multi_input_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )


# ---------------------------------------------------------------------------
# Backward-compatibility: MetaProcessor vs. legacy processors
# ---------------------------------------------------------------------------

class TestMetaProcessorBackwardCompatibility:
    """
    The MetaProcessor configured for a known task must produce the same set
    of output keys that the legacy processor + DataCollator produced.
    In the new design, ALL of these keys come from the MetaProcessor.
    """

    POSE2TEXT_EXPECTED_KEYS = {
        "input_frames",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    }

    TEXT2TEXT_EXPECTED_KEYS = {
        "input_ids",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    }

    def test_pose2text_produces_all_legacy_keys(
        self, tokenizer, pose_batch_samples
    ):
        meta = make_pose2text_meta(tokenizer)
        result = meta(pose_batch_samples)
        for key in self.POSE2TEXT_EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_text2text_produces_all_legacy_keys(
        self, tokenizer, text_batch_samples_no_signal
    ):
        meta = make_text2text_meta(tokenizer)
        result = meta(text_batch_samples_no_signal)
        for key in self.TEXT2TEXT_EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"


# ---------------------------------------------------------------------------
# DataCollator integration with MetaProcessor
# ---------------------------------------------------------------------------

class TestDataCollatorWithMetaProcessor:
    """
    In the new design, the DataCollator no longer needs a tokenizer — label
    processing happens inside the MetaProcessor's label slot (is_label=True).
    The DataCollator's responsibility shrinks to:
      1. Call processor(samples) to get the full batch dict.
      2. Optionally call model.prepare_decoder_input_ids_from_labels(labels).
    """

    def test_collator_can_be_instantiated_without_tokenizer(self, tokenizer):
        meta = make_pose2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        assert collator is not None

    def test_collator_output_contains_labels(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(pose_batch_samples)
        assert "labels" in result

    def test_collator_output_contains_input_frames(
        self, tokenizer, pose_batch_samples
    ):
        meta = make_pose2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(pose_batch_samples)
        assert "input_frames" in result

    def test_collator_batch_size_preserved(self, tokenizer, pose_batch_samples):
        meta = make_pose2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(pose_batch_samples)
        batch_size = len(pose_batch_samples)
        assert result["labels"].shape[0] == batch_size
        assert result["input_frames"].shape[0] == batch_size

    def test_collator_labels_come_from_meta_not_collator(
        self, tokenizer, pose_batch_samples
    ):
        """
        Regression: labels must be present even when the DataCollator is given
        no tokenizer, proving they originate from the MetaProcessor's label slot.
        """
        meta = make_pose2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta, tokenizer=None)
        result = collator(pose_batch_samples)
        assert "labels" in result
        assert isinstance(result["labels"], torch.Tensor)


# ---------------------------------------------------------------------------
# MultimodalMetaProcessor construction (flat slots)
# ---------------------------------------------------------------------------

class TestMetaProcessorConstruction:
    def test_slots_stored_in_order(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        assert len(meta.slots) == 4
        assert meta.slots[0].output_data_key == "input_frames"
        assert meta.slots[1].output_data_key == "labels"
        assert meta.slots[2].output_data_key == "encoder_prompt"
        assert meta.slots[3].output_data_key == "decoder_input_ids"

    def test_label_slot_is_labeled(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        label_slots = [s for s in meta.slots if s.is_label]
        assert len(label_slots) == 1
        assert label_slots[0].output_data_key == "labels"

    def test_tokenizer_stored(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        assert meta.tokenizer is tokenizer


# ---------------------------------------------------------------------------
# to_dict / serialization
# ---------------------------------------------------------------------------

class TestMetaProcessorSerialization:
    def test_to_dict_has_slots(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        d = meta.to_dict()
        assert "slots" in d
        assert len(d["slots"]) == 4

    def test_to_dict_slot_has_required_keys(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        d = meta.to_dict()
        for slot_dict in d["slots"]:
            assert "processor_class" in slot_dict
            assert "processor_kwargs" in slot_dict
            assert "output_data_key" in slot_dict
            assert "is_label" in slot_dict
            assert "column_map" in slot_dict

    def test_to_dict_label_slot_is_label_true(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        d = meta.to_dict()
        label_slots = [s for s in d["slots"] if s["is_label"]]
        assert len(label_slots) == 1

    def test_to_dict_is_json_serializable(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        d = meta.to_dict()
        json.dumps(d)  # should not raise


# ---------------------------------------------------------------------------
# save_pretrained / from_pretrained round-trip
# ---------------------------------------------------------------------------

class TestMetaProcessorSavePretrained:
    def test_roundtrip_slot_count(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_pretrained(tmpdir)
            loaded = MultimodalMetaProcessor.from_pretrained(tmpdir)
        assert len(loaded.slots) == len(meta.slots)

    def test_roundtrip_output_keys(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_pretrained(tmpdir)
            loaded = MultimodalMetaProcessor.from_pretrained(tmpdir)
        for orig, reloaded in zip(meta.slots, loaded.slots):
            assert orig.output_data_key == reloaded.output_data_key
            assert orig.output_mask_key == reloaded.output_mask_key
            assert orig.is_label == reloaded.is_label

    def test_roundtrip_column_maps(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_pretrained(tmpdir)
            loaded = MultimodalMetaProcessor.from_pretrained(tmpdir)
        for orig, reloaded in zip(meta.slots, loaded.slots):
            assert orig.column_map == reloaded.column_map

    def test_roundtrip_processor_classes(self, tokenizer):
        meta = _make_features2text_meta(tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_pretrained(tmpdir)
            loaded = MultimodalMetaProcessor.from_pretrained(tmpdir)
        for orig, reloaded in zip(meta.slots, loaded.slots):
            assert type(orig.processor).__name__ == type(reloaded.processor).__name__

    def test_roundtrip_output_consistent(self, tokenizer, features_batch_samples):
        """Loaded processor produces same output as original."""
        meta = _make_features2text_meta(tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta.save_pretrained(tmpdir)
            loaded = MultimodalMetaProcessor.from_pretrained(tmpdir)
        result_orig = meta(batch=features_batch_samples)
        result_loaded = loaded(batch=features_batch_samples)
        assert set(result_orig.keys()) == set(result_loaded.keys())
        assert torch.equal(result_orig["labels"], result_loaded["labels"])


# ---------------------------------------------------------------------------
# __call__ behaviour
# ---------------------------------------------------------------------------

class TestMetaProcessorCall:
    def test_returns_batch_feature(self, tokenizer, features_batch_samples):
        meta = _make_features2text_meta(tokenizer)
        result = meta(batch=features_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_has_expected_keys(self, tokenizer, features_batch_samples):
        meta = _make_features2text_meta(tokenizer)
        result = meta(batch=features_batch_samples)
        expected = {
            "input_frames",
            "attention_mask",
            "encoder_prompt",
            "encoder_prompt_length_padding_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
        for key in expected:
            assert key in result, f"Missing key: '{key}'"

    def test_batch_dimensions_consistent(self, tokenizer, features_batch_samples):
        meta = _make_features2text_meta(tokenizer)
        result = meta(batch=features_batch_samples)
        batch_size = len(features_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )

    def test_prepopulated_key_is_not_overwritten(self, tokenizer, features_batch_samples):
        """If output_data_key is already in batch_dict, slot is skipped."""
        meta = _make_features2text_meta(tokenizer)
        existing = torch.zeros(len(features_batch_samples), 5, 8)
        result = meta(batch=features_batch_samples, batch_dict={"input_frames": existing})
        assert torch.equal(result["input_frames"], existing)

    def test_labels_in_output(self, tokenizer, features_batch_samples):
        meta = _make_features2text_meta(tokenizer)
        result = meta(batch=features_batch_samples)
        assert "labels" in result


# ---------------------------------------------------------------------------
# _transform_get_items_output
# ---------------------------------------------------------------------------

class TestMetaProcessorTransform:
    def test_tensor_written_back(self, tokenizer, dummy_npy_file):
        meta = _make_features2text_meta(tokenizer)
        batch = {"signal": [dummy_npy_file]}
        result = meta._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)

    def test_text_not_corrupted(self, tokenizer, dummy_npy_file):
        """Text columns (no-op process_sample) must not be altered."""
        meta = _make_features2text_meta(tokenizer)
        original_prompt = "translate:"
        batch = {
            "signal": [dummy_npy_file],
            "encoder_prompt": [original_prompt],
        }
        result = meta._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)
        assert result["encoder_prompt"] == [original_prompt]

    def test_missing_primary_field_skipped(self, tokenizer):
        """If a slot's primary_field is absent from batch, it is silently skipped."""
        meta = _make_features2text_meta(tokenizer)
        batch = {"encoder_prompt": ["translate:"]}
        result = meta._transform_get_items_output(batch)
        assert result["encoder_prompt"] == ["translate:"]


# ---------------------------------------------------------------------------
# Multi-slot (multiple encoder inputs using features)
# ---------------------------------------------------------------------------

class TestMetaProcessorMultiSlot:
    def test_multi_input_produces_all_keys(self, tokenizer, features_batch_samples):
        meta = _make_multi_input_meta(tokenizer)
        result = meta(batch=features_batch_samples)
        assert "input_frames" in result
        assert "secondary_input" in result
        assert "encoder_prompt" in result
        assert "decoder_input_ids" in result


# ---------------------------------------------------------------------------
# Round-trip save / load — structural and behavioural equivalence
# ---------------------------------------------------------------------------

class TestMultimodalMetaProcessorRoundTrip:
    """
    Verifies that a MultimodalMetaProcessor saved with save_pretrained() and
    loaded with from_pretrained() is identical to the original — both in
    structure (slot configuration, processor types and kwargs) and in
    behaviour (identical output tensors for the same input batch).
    """

    def _assert_slots_equal(self, slot_a: ProcessorSlot, slot_b: ProcessorSlot):
        assert type(slot_a.processor) is type(slot_b.processor)
        assert slot_a.output_data_key == slot_b.output_data_key
        assert slot_a.output_mask_key == slot_b.output_mask_key
        assert slot_a.column_map == slot_b.column_map
        assert slot_a.is_label == slot_b.is_label

    def _assert_structure_equal(
        self, original: MultimodalMetaProcessor, loaded: MultimodalMetaProcessor
    ):
        assert type(loaded) is type(original)
        assert len(loaded.slots) == len(original.slots)
        for s_orig, s_load in zip(original.slots, loaded.slots):
            self._assert_slots_equal(s_orig, s_load)

    # ------------------------------------------------------------------
    # text→text (no external files needed — simplest round-trip)
    # ------------------------------------------------------------------

    def test_loaded_is_multimodal_meta_processor(self, tokenizer, tmp_path, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))
        assert isinstance(loaded, MultimodalMetaProcessor)

    def test_text2text_slot_structure_preserved(self, tokenizer, tmp_path):
        meta = make_text2text_meta(tokenizer)
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))
        self._assert_structure_equal(meta, loaded)

    def test_text2text_encoder_slot_processor_type(self, tokenizer, tmp_path):
        meta = make_text2text_meta(tokenizer)
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))
        encoder_slots = [s for s in loaded.slots if not s.is_label]
        assert isinstance(encoder_slots[0].processor, TextModalityProcessor)

    def test_text2text_output_identical(self, tokenizer, tmp_path, text_batch_samples_no_signal):
        meta = make_text2text_meta(tokenizer)
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))

        result_orig = meta(text_batch_samples_no_signal)
        result_load = loaded(text_batch_samples_no_signal)

        for key in result_orig:
            assert key in result_load, f"Key '{key}' missing from loaded output"
            if isinstance(result_orig[key], torch.Tensor):
                assert torch.equal(result_orig[key], result_load[key]), (
                    f"Tensor mismatch for key '{key}'"
                )

    def test_text2text_transform_output_identical(self, tokenizer, tmp_path):
        meta = make_text2text_meta(tokenizer)
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))

        batch = {
            "signal": ["Hello world", "Good morning"],
            "encoder_prompt": ["translate:", "translate:"],
            "decoder_prompt": ["de:", "de:"],
            "output": ["Hallo Welt", "Guten Morgen"],
        }
        result_orig = meta._transform_get_items_output(batch.copy())
        result_load = loaded._transform_get_items_output(batch.copy())

        for key in result_orig:
            for v_orig, v_load in zip(result_orig[key], result_load[key]):
                if isinstance(v_orig, torch.Tensor):
                    assert torch.equal(v_orig, v_load), (
                        f"Mismatch in _transform_get_items_output for key '{key}'"
                    )

    # ------------------------------------------------------------------
    # features→text (non-trivial ModalityProcessor kwargs)
    # ------------------------------------------------------------------

    def _make_features2text_meta_with_kwargs(
        self, tokenizer, skip_frames_stride=2, temporal_dimention_position=1
    ):
        return MultimodalMetaProcessor(
            slots=[
                ProcessorSlot(
                    processor=FeaturesModalityProcessor(
                        skip_frames_stride=skip_frames_stride,
                        temporal_dimention_position=temporal_dimention_position,
                        use_cache=False,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="target"),
                    output_data_key="labels",
                    is_label=True,
                    column_map={"decoder_prompt": "target_prefix", "output": "target"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                    output_data_key="encoder_prompt",
                    output_mask_key="encoder_prompt_length_padding_mask",
                    column_map={"encoder_prompt": "signal"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="input"),
                    output_data_key="decoder_input_ids",
                    output_mask_key="decoder_attention_mask",
                    column_map={"decoder_prompt": "signal"},
                ),
            ],
            tokenizer=tokenizer,
        )

    def test_features2text_processor_kwargs_preserved(self, tokenizer, tmp_path):
        """Non-trivial ModalityProcessor kwargs must survive the save/load cycle."""
        meta = self._make_features2text_meta_with_kwargs(
            tokenizer, skip_frames_stride=3, temporal_dimention_position=1
        )
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))

        orig_proc = meta.slots[0].processor
        load_proc = loaded.slots[0].processor

        assert isinstance(load_proc, FeaturesModalityProcessor)
        assert load_proc.skip_frames_stride == orig_proc.skip_frames_stride
        assert load_proc.temporal_dimention_position == orig_proc.temporal_dimention_position
        assert load_proc.use_cache == orig_proc.use_cache

    def test_features2text_output_identical(self, tokenizer, tmp_path, features_batch_samples):
        meta = self._make_features2text_meta_with_kwargs(tokenizer)
        meta.save_pretrained(str(tmp_path))
        loaded = MultimodalMetaProcessor.from_pretrained(str(tmp_path))

        result_orig = meta(features_batch_samples)
        result_load = loaded(features_batch_samples)

        for key in result_orig:
            assert key in result_load, f"Key '{key}' missing from loaded output"
            if isinstance(result_orig[key], torch.Tensor):
                assert torch.equal(result_orig[key], result_load[key]), (
                    f"Tensor mismatch for key '{key}'"
                )
