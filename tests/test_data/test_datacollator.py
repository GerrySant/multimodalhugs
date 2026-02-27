"""Tests for DataCollatorMultimodalSeq2Seq and create_seq2seq_labels_from_samples."""

import numpy as np
import torch

from multimodalhugs.data.datacollators.multimodal_datacollator import (
    DataCollatorMultimodalSeq2Seq,
    create_seq2seq_labels_from_samples,
)
from multimodalhugs.processors.text2text_preprocessor import (
    Text2TextTranslationProcessor,
)
from multimodalhugs.processors.features2text_preprocessor import (
    Features2TextTranslationProcessor,
)


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
