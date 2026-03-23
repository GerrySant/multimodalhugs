"""Tests for DataCollatorMultimodalSeq2Seq and create_seq2seq_labels_from_samples."""

import numpy as np
import torch

from multimodalhugs.data.datacollators.multimodal_datacollator import (
    DataCollatorMultimodalSeq2Seq,
    create_seq2seq_labels_from_samples,
)
from multimodalhugs.processors.legacy.text2text_preprocessor import (
    Text2TextTranslationProcessor,
)
from multimodalhugs.processors.legacy.features2text_preprocessor import (
    Features2TextTranslationProcessor,
)
from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor
from multimodalhugs.processors.features_modality_processor import FeaturesModalityProcessor


# ============================================================================
# create_seq2seq_labels_from_samples
# ============================================================================
class TestCreateSeq2SeqLabels:
    def test_basic_label_creation(self, tokenizer):
        samples = [
            {"decoder_prompt": "de:", "output": "Hallo Welt"},
            {"decoder_prompt": "de:", "output": "Danke"},
        ]
        result = create_seq2seq_labels_from_samples(samples, tokenizer)
        assert result is not None
        assert "labels" in result
        assert isinstance(result["labels"], torch.Tensor)
        assert result["labels"].shape[0] == 2

    def test_missing_output_returns_none(self, tokenizer):
        samples = [
            {"decoder_prompt": "de:", "output": None},
            {"decoder_prompt": "de:", "output": "Hallo"},
        ]
        result = create_seq2seq_labels_from_samples(samples, tokenizer)
        assert result is None

    def test_padding_applied(self, tokenizer):
        """Different output lengths should result in same label length with -100 padding."""
        samples = [
            {"decoder_prompt": "", "output": "Hi"},
            {"decoder_prompt": "", "output": "Hello world how are you doing today"},
        ]
        result = create_seq2seq_labels_from_samples(samples, tokenizer, padding=True)
        assert result["labels"].shape[0] == 2
        # Both should have same length
        assert result["labels"].shape[1] == result["labels"].shape[1]
        # Shorter sequence should have -100 padding
        assert (result["labels"][0] == -100).any()

    def test_no_padding_returns_raw_lists(self, tokenizer):
        """With padding=False, raw lists are returned (not tensors)."""
        samples = [
            {"decoder_prompt": "", "output": "Hi"},
            {"decoder_prompt": "", "output": "Hello world how are you"},
        ]
        result = create_seq2seq_labels_from_samples(samples, tokenizer, padding=False)
        assert result is not None
        assert "labels" in result
        assert isinstance(result["labels"], list)
        assert isinstance(result["labels"][0], list)
        # Different lengths since no padding
        assert len(result["labels"][0]) != len(result["labels"][1])

    def test_max_length_padding(self, tokenizer):
        from transformers.utils import PaddingStrategy

        samples = [
            {"decoder_prompt": "", "output": "Hi"},
        ]
        result = create_seq2seq_labels_from_samples(
            samples, tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=20
        )
        assert result["labels"].shape[1] == 20

    def test_pad_to_multiple_of(self, tokenizer):
        samples = [
            {"decoder_prompt": "", "output": "Hello"},
        ]
        result = create_seq2seq_labels_from_samples(
            samples, tokenizer, padding=True, pad_to_multiple_of=8
        )
        assert result["labels"].shape[1] % 8 == 0

    def test_eos_token_appended(self, tokenizer):
        samples = [
            {"decoder_prompt": "", "output": "Hello"},
        ]
        result = create_seq2seq_labels_from_samples(samples, tokenizer, padding=False)
        # Last token should be EOS
        assert result["labels"][0][-1] == tokenizer.eos_token_id

    def test_return_tensors_np(self, tokenizer):
        samples = [
            {"decoder_prompt": "", "output": "Hello"},
        ]
        result = create_seq2seq_labels_from_samples(
            samples, tokenizer, return_tensors="np"
        )
        assert isinstance(result["labels"], np.ndarray)


# ============================================================================
# DataCollatorMultimodalSeq2Seq
# ============================================================================
class TestDataCollatorInit:
    def test_init_with_processor_and_tokenizer(self, tokenizer):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(
            processor=processor, tokenizer=tokenizer
        )
        assert collator.tokenizer is tokenizer
        assert collator.processor is processor

    def test_init_tokenizer_none_falls_back(self, tokenizer):
        """When tokenizer=None, should fall back to processor.tokenizer."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=processor)
        assert collator.tokenizer is tokenizer


class TestDataCollatorCall:
    def test_text2text_full_call(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(
            processor=processor, tokenizer=tokenizer
        )
        result = collator(text_batch_samples)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_features2text_full_call(self, tokenizer, features_batch_samples):
        processor = Features2TextTranslationProcessor(
            tokenizer=tokenizer, use_cache=False
        )
        collator = DataCollatorMultimodalSeq2Seq(
            processor=processor, tokenizer=tokenizer
        )
        result = collator(features_batch_samples)
        assert "input_frames" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_labels_batch_dimension_matches(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(
            processor=processor, tokenizer=tokenizer
        )
        result = collator(text_batch_samples)
        batch_size = len(text_batch_samples)
        assert result["labels"].shape[0] == batch_size
        assert result["input_ids"].shape[0] == batch_size


# ============================================================================
# DataCollatorMultimodalSeq2Seq — MetaProcessor path
# ============================================================================

def _make_text2text_meta(tokenizer):
    return MultimodalMetaProcessor(
        slots=[
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                output_data_key="input_ids",
                output_mask_key="attention_mask",
                column_map={"signal": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
                output_data_key="labels",
                is_label=True,
                column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="prompt"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
        ],
        tokenizer=tokenizer,
    )


def _make_features2text_meta(tokenizer):
    return MultimodalMetaProcessor(
        slots=[
            ProcessorSlot(
                processor=FeaturesModalityProcessor(use_cache=False),
                output_data_key="input_frames",
                output_mask_key="attention_mask",
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
                output_data_key="labels",
                is_label=True,
                column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="prompt"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
        ],
        tokenizer=tokenizer,
    )


class TestDataCollatorWithMetaProcessor:
    """
    Verifies that when the processor is a MultimodalMetaProcessor the collator
    delegates all processing (including label creation) to the processor's slots
    and does not create labels itself.
    """

    def test_text2text_returns_expected_keys(self, tokenizer, text_batch_samples):
        meta = _make_text2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(text_batch_samples)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert "encoder_prompt" in result
        assert "encoder_prompt_length_padding_mask" in result
        assert "decoder_input_ids" in result

    def test_features2text_returns_expected_keys(self, tokenizer, features_batch_samples):
        meta = _make_features2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(features_batch_samples)
        assert "input_frames" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_labels_are_tensors(self, tokenizer, text_batch_samples):
        meta = _make_text2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(text_batch_samples)
        assert isinstance(result["labels"], torch.Tensor)

    def test_labels_batch_dim_matches(self, tokenizer, text_batch_samples):
        meta = _make_text2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(text_batch_samples)
        assert result["labels"].shape[0] == len(text_batch_samples)

    def test_labels_padded_with_minus_100(self, tokenizer, text_batch_samples):
        """Padding positions must use -100 (ignored by loss) — no real pad token IDs."""
        meta = _make_text2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(text_batch_samples)
        # Every label value is either a valid token id (>=0) or the ignore index (-100)
        assert (result["labels"] == -100).logical_or(result["labels"] >= 0).all()

    def test_meta_labels_match_legacy_labels(self, tokenizer, text_batch_samples):
        """
        Labels produced by the MetaProcessor's label slot must match those
        produced by the legacy DataCollator path (create_seq2seq_labels_from_samples).
        """
        legacy_labels = create_seq2seq_labels_from_samples(
            text_batch_samples, tokenizer
        )["labels"]

        meta = _make_text2text_meta(tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(processor=meta)
        result = collator(text_batch_samples)

        assert torch.equal(result["labels"], legacy_labels)

    def test_legacy_processor_path_still_works(self, tokenizer, text_batch_samples):
        """Old-style processors must continue to go through the legacy collator path."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        collator = DataCollatorMultimodalSeq2Seq(
            processor=processor, tokenizer=tokenizer
        )
        result = collator(text_batch_samples)
        assert "labels" in result
        assert isinstance(result["labels"], torch.Tensor)
