"""Tests for build_processor_from_config in setup_utils."""

import pytest
from omegaconf import OmegaConf

from multimodalhugs.training_setup.setup_utils import build_processor_from_config
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
