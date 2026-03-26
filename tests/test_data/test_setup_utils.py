"""Tests for build_processor_from_config and expand_pipeline_shorthand in setup_utils."""

import pytest
from omegaconf import OmegaConf

from multimodalhugs.training_setup.setup_utils import (
    build_processor_from_config,
    expand_pipeline_shorthand,
)
from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor
from multimodalhugs.processors.features_modality_processor import FeaturesModalityProcessor

from tests.test_data.conftest import TINY_TOKENIZER_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(slots_yaml: str):
    """Wrap a YAML slots block in a processor: section and return OmegaConf."""
    raw = f"""
processor:
  slots:
{slots_yaml}
"""
    cfg = OmegaConf.create(raw)
    return cfg.processor


def _minimal_text_slots_yaml():
    return f"""\
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: input
        tokenizer_path: {TINY_TOKENIZER_PATH}
      output_data_key: input_ids
      output_mask_key: attention_mask
      column_map:
        signal: signal
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: target
        tokenizer_path: {TINY_TOKENIZER_PATH}
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix
        output: target
"""


# ---------------------------------------------------------------------------
# TestBuildProcessorFromConfig — unit tests
# ---------------------------------------------------------------------------

class TestBuildProcessorFromConfigReturnsNone:
    """Returns None when no slots key is present."""

    def test_returns_none_when_slots_absent(self):
        cfg = OmegaConf.create({"text_tokenizer_path": TINY_TOKENIZER_PATH})
        assert build_processor_from_config(cfg) is None

    def test_returns_none_when_slots_empty_list(self):
        cfg = OmegaConf.create({"slots": []})
        assert build_processor_from_config(cfg) is None

    def test_returns_none_for_none_cfg(self):
        assert build_processor_from_config(None) is None


class TestBuildProcessorFromConfigReturnsProcessor:
    """Returns a correctly constructed MultimodalMetaProcessor when slots are declared."""

    def test_returns_meta_processor(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        assert isinstance(proc, MultimodalMetaProcessor)

    def test_slot_count_matches(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        assert len(proc.slots) == 2

    def test_output_data_keys(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        keys = [s.output_data_key for s in proc.slots]
        assert "input_ids" in keys
        assert "labels" in keys

    def test_output_mask_key(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        encoder_slot = next(s for s in proc.slots if s.output_data_key == "input_ids")
        assert encoder_slot.output_mask_key == "attention_mask"

    def test_is_label_flag(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        label_slot = next(s for s in proc.slots if s.output_data_key == "labels")
        assert label_slot.is_label is True

    def test_non_label_slot_is_false(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        encoder_slot = next(s for s in proc.slots if s.output_data_key == "input_ids")
        assert encoder_slot.is_label is False

    def test_column_map_set(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        label_slot = next(s for s in proc.slots if s.output_data_key == "labels")
        assert label_slot.column_map == {"decoder_prompt": "target_prefix", "output": "target"}

    def test_processor_class_instantiated(self):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        for slot in proc.slots:
            assert isinstance(slot.processor, TextModalityProcessor)

    def test_tokenizer_loaded_from_path(self):
        """TextModalityProcessor loads its own tokenizer from tokenizer_path."""
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        for slot in proc.slots:
            assert slot.processor.tokenizer is not None

    def test_meta_processor_tokenizer_auto_derived(self):
        """MultimodalMetaProcessor.tokenizer is derived from the first text slot."""
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg)
        assert proc.tokenizer is not None


class TestBuildProcessorFromConfigFeaturesSlot:
    """FeaturesModalityProcessor (no tokenizer param) is built without error."""

    def test_features_slot_built(self):
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      processor_kwargs:
        use_cache: false
      output_data_key: input_frames
      output_mask_key: attention_mask
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg)
        assert len(proc.slots) == 1
        assert isinstance(proc.slots[0].processor, FeaturesModalityProcessor)

    def test_features_meta_tokenizer_is_none(self):
        """No text slots → MultimodalMetaProcessor.tokenizer is None."""
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      processor_kwargs:
        use_cache: false
      output_data_key: input_frames
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg)
        assert proc.tokenizer is None

    def test_features_processor_kwargs_forwarded(self):
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      processor_kwargs:
        use_cache: false
        skip_frames_stride: 2
      output_data_key: input_frames
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg)
        assert proc.slots[0].processor.skip_frames_stride == 2


class TestBuildProcessorFromConfigDefaultColumnMap:
    """A slot without an explicit column_map gets the default {"signal": "signal"}."""

    def test_default_column_map(self):
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      output_data_key: input_frames
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg)
        assert proc.slots[0].column_map == {"signal": "signal"}


class TestBuildProcessorFromConfigProducesValidOutput:
    """End-to-end: the built processor processes a text batch correctly."""

    def test_text_batch_produces_expected_keys(self, text_batch_samples):
        slots_yaml = f"""\
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: input
        tokenizer_path: {TINY_TOKENIZER_PATH}
      output_data_key: input_ids
      output_mask_key: attention_mask
      column_map:
        signal: signal
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: target
        tokenizer_path: {TINY_TOKENIZER_PATH}
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix
        output: target
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg)
        result = proc(batch=text_batch_samples)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_text_batch_label_batch_dim(self, text_batch_samples):
        slots_yaml = f"""\
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: input
        tokenizer_path: {TINY_TOKENIZER_PATH}
      output_data_key: input_ids
      output_mask_key: attention_mask
      column_map:
        signal: signal
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: target
        tokenizer_path: {TINY_TOKENIZER_PATH}
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix
        output: target
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg)
        result = proc(batch=text_batch_samples)
        assert result["labels"].shape[0] == len(text_batch_samples)


# ---------------------------------------------------------------------------
# Helpers shared by expand_pipeline_shorthand tests
# ---------------------------------------------------------------------------

def _make_pipeline_cfg(extra: dict = None):
    """Return a minimal valid pipeline shorthand config as OmegaConf."""
    base = {"pipeline": "pose2text", "tokenizer_path": TINY_TOKENIZER_PATH}
    if extra:
        base.update(extra)
    return OmegaConf.create(base)


def _make_pipeline_dict(extra: dict = None):
    """Same as _make_pipeline_cfg but returns a plain dict (not OmegaConf)."""
    base = {"pipeline": "pose2text", "tokenizer_path": TINY_TOKENIZER_PATH}
    if extra:
        base.update(extra)
    return base


# Standard output_data_keys produced by any *2text pipeline shorthand.
_STANDARD_KEYS = {"input_frames", "labels", "encoder_prompt", "decoder_input_ids"}


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandPassthrough — non-shorthand configs unchanged
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandPassthrough:
    """Configs without a pipeline: key are returned unchanged."""

    def test_passthrough_none(self):
        assert expand_pipeline_shorthand(None) is None

    def test_passthrough_slots_present(self):
        cfg = OmegaConf.create({"slots": [{"processor_class": "FeaturesModalityProcessor",
                                            "output_data_key": "input_frames"}]})
        result = expand_pipeline_shorthand(cfg)
        # Identical object — no expansion occurred.
        assert result is cfg

    def test_passthrough_legacy_config(self):
        cfg = OmegaConf.create({"text_tokenizer_path": TINY_TOKENIZER_PATH})
        result = expand_pipeline_shorthand(cfg)
        assert result is cfg

    def test_passthrough_preserves_omegaconf_type(self):
        cfg = OmegaConf.create({"slots": []})
        result = expand_pipeline_shorthand(cfg)
        assert OmegaConf.is_config(result)


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandValidation — error cases
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandValidation:
    """Invalid shorthand configs raise informative errors."""

    def test_unknown_pipeline_raises(self):
        cfg = OmegaConf.create({"pipeline": "unknown2text",
                                 "tokenizer_path": TINY_TOKENIZER_PATH})
        with pytest.raises(ValueError, match="Unknown pipeline"):
            expand_pipeline_shorthand(cfg)

    def test_missing_tokenizer_path_raises(self):
        cfg = OmegaConf.create({"pipeline": "pose2text"})
        with pytest.raises(ValueError, match="tokenizer_path"):
            expand_pipeline_shorthand(cfg)


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandStructure — slot list shape and keys
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandStructure:
    """Expanded config has the expected slots list structure."""

    def test_returns_slots_key(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        assert "slots" in (result if isinstance(result, dict) else OmegaConf.to_container(result))

    def test_four_slots_generated(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        slots = OmegaConf.to_container(result)["slots"]
        assert len(slots) == 4

    def test_output_data_keys_match_standard(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        keys = {s["output_data_key"] for s in OmegaConf.to_container(result)["slots"]}
        assert keys == _STANDARD_KEYS

    def test_pipeline_key_removed(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        assert "pipeline" not in OmegaConf.to_container(result)

    def test_shorthand_keys_removed(self):
        cfg = _make_pipeline_cfg({"new_vocabulary": "__asl__",
                                   "modality_kwargs": {"skip_frames_stride": 2}})
        result = expand_pipeline_shorthand(cfg)
        d = OmegaConf.to_container(result)
        for key in ("pipeline", "tokenizer_path", "new_vocabulary",
                    "modality_kwargs", "slot_overrides"):
            assert key not in d, f"Shorthand key '{key}' should have been removed"

    def test_returns_omegaconf_for_omegaconf_input(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        assert OmegaConf.is_config(result)

    def test_returns_dict_for_dict_input(self):
        result = expand_pipeline_shorthand(_make_pipeline_dict())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandModalitySlot — first slot per pipeline
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandModalitySlot:
    """The modality (first) slot has the correct processor class per pipeline."""

    @pytest.mark.parametrize("pipeline,expected_class", [
        ("pose2text",         "PoseModalityProcessor"),
        ("video2text",        "VideoModalityProcessor"),
        ("features2text",     "FeaturesModalityProcessor"),
        ("image2text",        "ImageModalityProcessor"),
        ("signwriting2text",  "SignwritingModalityProcessor"),
        ("text2text",         "TextModalityProcessor"),
    ])
    def test_modality_processor_class(self, pipeline, expected_class):
        cfg = OmegaConf.create({"pipeline": pipeline,
                                  "tokenizer_path": TINY_TOKENIZER_PATH})
        result = expand_pipeline_shorthand(cfg)
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert first_slot["processor_class"] == expected_class

    def test_modality_slot_output_data_key(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert first_slot["output_data_key"] == "input_frames"

    def test_modality_slot_output_mask_key(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert first_slot["output_mask_key"] == "attention_mask"

    def test_pose_column_map_has_offsets(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert "signal_start" in first_slot["column_map"]
        assert "signal_end" in first_slot["column_map"]

    def test_image_column_map_no_offsets(self):
        cfg = OmegaConf.create({"pipeline": "image2text",
                                  "tokenizer_path": TINY_TOKENIZER_PATH})
        result = expand_pipeline_shorthand(cfg)
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert "signal_start" not in first_slot["column_map"]

    def test_modality_kwargs_forwarded(self):
        cfg = _make_pipeline_cfg({"modality_kwargs": {"reduce_holistic_poses": True}})
        result = expand_pipeline_shorthand(cfg)
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert first_slot.get("processor_kwargs", {}).get("reduce_holistic_poses") is True

    def test_no_processor_kwargs_when_modality_kwargs_absent(self):
        result = expand_pipeline_shorthand(_make_pipeline_cfg())
        first_slot = OmegaConf.to_container(result)["slots"][0]
        # No modality_kwargs → processor_kwargs should be absent (or empty).
        assert not first_slot.get("processor_kwargs")

    def test_text2text_modality_slot_has_tokenizer(self):
        cfg = OmegaConf.create({"pipeline": "text2text",
                                  "tokenizer_path": TINY_TOKENIZER_PATH})
        result = expand_pipeline_shorthand(cfg)
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert first_slot["processor_kwargs"]["tokenizer_path"] == TINY_TOKENIZER_PATH

    def test_text2text_modality_slot_has_role_input(self):
        cfg = OmegaConf.create({"pipeline": "text2text",
                                  "tokenizer_path": TINY_TOKENIZER_PATH})
        result = expand_pipeline_shorthand(cfg)
        first_slot = OmegaConf.to_container(result)["slots"][0]
        assert first_slot["processor_kwargs"]["role"] == "input"


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandTextSlots — shared text output slots
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandTextSlots:
    """The three standard text output slots are wired correctly."""

    def _get_slots_by_key(self, cfg):
        result = expand_pipeline_shorthand(cfg)
        slots = OmegaConf.to_container(result)["slots"]
        return {s["output_data_key"]: s for s in slots}

    def test_labels_slot_is_label_true(self):
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        assert slots["labels"]["is_label"] is True

    def test_labels_slot_column_map(self):
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        assert slots["labels"]["column_map"] == {
            "decoder_prompt": "target_prefix",
            "output": "target",
        }

    def test_labels_slot_role_target(self):
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        assert slots["labels"]["processor_kwargs"]["role"] == "target"

    def test_encoder_prompt_slot_has_mask_key(self):
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        assert slots["encoder_prompt"]["output_mask_key"] == "encoder_prompt_length_padding_mask"

    def test_decoder_input_ids_slot_has_mask_key(self):
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        assert slots["decoder_input_ids"]["output_mask_key"] == "decoder_attention_mask"

    def test_tokenizer_path_in_text_slots(self):
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        for key in ("labels", "encoder_prompt", "decoder_input_ids"):
            assert slots[key]["processor_kwargs"]["tokenizer_path"] == TINY_TOKENIZER_PATH

    def test_new_vocabulary_propagated(self):
        cfg = _make_pipeline_cfg({"new_vocabulary": "__asl__"})
        slots = self._get_slots_by_key(cfg)
        for key in ("labels", "encoder_prompt", "decoder_input_ids"):
            assert slots[key]["processor_kwargs"]["new_vocabulary"] == "__asl__"

    def test_new_vocabulary_absent_when_not_set(self):
        # new_vocabulary is optional; omitting it means it is not injected.
        slots = self._get_slots_by_key(_make_pipeline_cfg())
        for key in ("labels", "encoder_prompt", "decoder_input_ids"):
            assert "new_vocabulary" not in slots[key]["processor_kwargs"]


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandSlotOverrides — per-slot sparse overrides
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandSlotOverrides:
    """slot_overrides merges correctly into the generated slots."""

    def test_override_column_map_merged(self):
        cfg = _make_pipeline_cfg({
            "slot_overrides": {"encoder_prompt": {"column_map": {"my_col": "signal"}}}
        })
        result = expand_pipeline_shorthand(cfg)
        slots_by_key = {s["output_data_key"]: s
                        for s in OmegaConf.to_container(result)["slots"]}
        # The override key is present...
        assert "my_col" in slots_by_key["encoder_prompt"]["column_map"]
        # ...and the original key is preserved (shallow merge, not replace).
        assert "encoder_prompt" in slots_by_key["encoder_prompt"]["column_map"]

    def test_override_scalar_replaces(self):
        cfg = _make_pipeline_cfg({
            "slot_overrides": {"encoder_prompt": {"output_mask_key": "custom_mask"}}
        })
        result = expand_pipeline_shorthand(cfg)
        slots_by_key = {s["output_data_key"]: s
                        for s in OmegaConf.to_container(result)["slots"]}
        assert slots_by_key["encoder_prompt"]["output_mask_key"] == "custom_mask"

    def test_override_processor_kwargs_merged(self):
        cfg = _make_pipeline_cfg({
            "slot_overrides": {"labels": {"processor_kwargs": {"pad_token_id": 1}}}
        })
        result = expand_pipeline_shorthand(cfg)
        slots_by_key = {s["output_data_key"]: s
                        for s in OmegaConf.to_container(result)["slots"]}
        # Override key added...
        assert slots_by_key["labels"]["processor_kwargs"]["pad_token_id"] == 1
        # ...existing keys preserved.
        assert "tokenizer_path" in slots_by_key["labels"]["processor_kwargs"]

    def test_unknown_override_key_is_ignored(self, caplog):
        import logging
        cfg = _make_pipeline_cfg({
            "slot_overrides": {"nonexistent_key": {"column_map": {"x": "y"}}}
        })
        with caplog.at_level(logging.WARNING):
            expand_pipeline_shorthand(cfg)  # must not raise
        assert "nonexistent_key" in caplog.text


# ---------------------------------------------------------------------------
# TestExpandPipelineShorthandEndToEnd — roundtrip through build_processor
# ---------------------------------------------------------------------------

class TestExpandPipelineShorthandEndToEnd:
    """Shorthand → expand → build_processor_from_config produces a working processor."""

    def test_pipeline_shorthand_builds_processor(self):
        cfg = OmegaConf.create({
            "pipeline": "features2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        })
        proc = build_processor_from_config(cfg)
        assert isinstance(proc, MultimodalMetaProcessor)

    def test_pipeline_shorthand_slot_count(self):
        cfg = OmegaConf.create({
            "pipeline": "features2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        })
        proc = build_processor_from_config(cfg)
        assert len(proc.slots) == 4

    def test_pipeline_shorthand_tokenizer_set(self):
        cfg = OmegaConf.create({
            "pipeline": "features2text",
            "tokenizer_path": TINY_TOKENIZER_PATH,
        })
        proc = build_processor_from_config(cfg)
        assert proc.tokenizer is not None
