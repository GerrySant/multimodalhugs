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
  text_tokenizer_path: {TINY_TOKENIZER_PATH}
  slots:
{slots_yaml}
"""
    cfg = OmegaConf.create(raw)
    return cfg.processor


def _minimal_text_slots_yaml():
    return """\
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: encoder
      output_data_key: input_ids
      output_mask_key: attention_mask
      column_map:
        signal: signal
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: label
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: decoder_prompt
        output: output
"""


# ---------------------------------------------------------------------------
# TestBuildProcessorFromConfig — unit tests
# ---------------------------------------------------------------------------

class TestBuildProcessorFromConfigReturnsNone:
    """Returns None when no slots key is present."""

    def test_returns_none_when_slots_absent(self, tokenizer):
        cfg = OmegaConf.create({"text_tokenizer_path": TINY_TOKENIZER_PATH})
        assert build_processor_from_config(cfg, tokenizer) is None

    def test_returns_none_when_slots_empty_list(self, tokenizer):
        cfg = OmegaConf.create({"slots": []})
        assert build_processor_from_config(cfg, tokenizer) is None

    def test_returns_none_for_none_cfg(self, tokenizer):
        assert build_processor_from_config(None, tokenizer) is None


class TestBuildProcessorFromConfigReturnsProcessor:
    """Returns a correctly constructed MultimodalMetaProcessor when slots are declared."""

    def test_returns_meta_processor(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        assert isinstance(proc, MultimodalMetaProcessor)

    def test_slot_count_matches(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        assert len(proc.slots) == 2

    def test_output_data_keys(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        keys = [s.output_data_key for s in proc.slots]
        assert "input_ids" in keys
        assert "labels" in keys

    def test_output_mask_key(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        encoder_slot = next(s for s in proc.slots if s.output_data_key == "input_ids")
        assert encoder_slot.output_mask_key == "attention_mask"

    def test_is_label_flag(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        label_slot = next(s for s in proc.slots if s.output_data_key == "labels")
        assert label_slot.is_label is True

    def test_non_label_slot_is_false(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        encoder_slot = next(s for s in proc.slots if s.output_data_key == "input_ids")
        assert encoder_slot.is_label is False

    def test_column_map_set(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        label_slot = next(s for s in proc.slots if s.output_data_key == "labels")
        assert label_slot.column_map == {"decoder_prompt": "decoder_prompt", "output": "output"}

    def test_processor_class_instantiated(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        for slot in proc.slots:
            assert isinstance(slot.processor, TextModalityProcessor)

    def test_tokenizer_injected_automatically(self, tokenizer):
        """TextModalityProcessor receives the tokenizer even if not in processor_kwargs."""
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        for slot in proc.slots:
            assert slot.processor.tokenizer is tokenizer

    def test_meta_processor_tokenizer_set(self, tokenizer):
        cfg = _make_cfg(_minimal_text_slots_yaml())
        proc = build_processor_from_config(cfg, tokenizer)
        assert proc.tokenizer is tokenizer


class TestBuildProcessorFromConfigFeaturesSlot:
    """FeaturesModalityProcessor (no tokenizer param) is built without error."""

    def test_features_slot_no_tokenizer_injected(self, tokenizer):
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      processor_kwargs:
        use_cache: false
      output_data_key: input_frames
      output_mask_key: attention_mask
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg, tokenizer)
        assert len(proc.slots) == 1
        assert isinstance(proc.slots[0].processor, FeaturesModalityProcessor)

    def test_features_processor_kwargs_forwarded(self, tokenizer):
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      processor_kwargs:
        use_cache: false
        skip_frames_stride: 2
      output_data_key: input_frames
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg, tokenizer)
        assert proc.slots[0].processor.skip_frames_stride == 2


class TestBuildProcessorFromConfigDefaultColumnMap:
    """A slot without an explicit column_map gets the default {"signal": "signal"}."""

    def test_default_column_map(self, tokenizer):
        slots_yaml = """\
    - processor_class: FeaturesModalityProcessor
      output_data_key: input_frames
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg, tokenizer)
        assert proc.slots[0].column_map == {"signal": "signal"}


class TestBuildProcessorFromConfigProducesValidOutput:
    """End-to-end: the built processor processes a text batch correctly."""

    def test_text_batch_produces_expected_keys(self, tokenizer, text_batch_samples):
        slots_yaml = """\
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: encoder
      output_data_key: input_ids
      output_mask_key: attention_mask
      column_map:
        signal: signal
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: label
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: decoder_prompt
        output: output
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg, tokenizer)
        result = proc(batch=text_batch_samples)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_text_batch_label_batch_dim(self, tokenizer, text_batch_samples):
        slots_yaml = """\
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: encoder
      output_data_key: input_ids
      output_mask_key: attention_mask
      column_map:
        signal: signal
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: label
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: decoder_prompt
        output: output
"""
        cfg = _make_cfg(slots_yaml)
        proc = build_processor_from_config(cfg, tokenizer)
        result = proc(batch=text_batch_samples)
        assert result["labels"].shape[0] == len(text_batch_samples)
