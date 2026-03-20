"""Tests for Text2TextTranslationProcessor."""

import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.text2text_preprocessor import (
    Text2TextTranslationProcessor,
)


class TestText2TextObtainMultimodalInputAndMasks:
    def test_returns_input_ids_and_attention_mask(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_shapes_are_correct(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        batch_size = len(text_batch_samples)
        assert result["input_ids"].shape[0] == batch_size
        assert result["attention_mask"].shape[0] == batch_size
        assert result["input_ids"].shape == result["attention_mask"].shape

    def test_padding_works_for_variable_length(self, tokenizer):
        """Variable-length text inputs should be padded to the same length."""
        batch = [
            {"signal": "Hi", "encoder_prompt": "", "decoder_prompt": "", "output": "x"},
            {
                "signal": "Hello world how are you doing today",
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "y",
            },
        ]
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=batch)
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["attention_mask"].shape[1]


class TestText2TextPromptSlots:
    """Verify encoder/decoder prompt processing via the slot-based design."""

    def test_encoder_prompt_keys_and_shape(self, tokenizer, text_batch_samples):
        """encoder_prompt_slot should produce encoder_prompt and its padding mask."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        assert "encoder_prompt" in result
        assert "encoder_prompt_length_padding_mask" in result
        assert result["encoder_prompt"].shape[0] == len(text_batch_samples)

    def test_decoder_prompt_keys_and_shape(self, tokenizer, text_batch_samples):
        """decoder_prompt_slot should produce decoder_input_ids and its attention mask."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        assert "decoder_input_ids" in result
        assert "decoder_attention_mask" in result
        assert result["decoder_input_ids"].shape[0] == len(text_batch_samples)


class TestText2TextProcessorCall:
    def test_full_call_returns_batch_feature(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        assert isinstance(result, BatchFeature)

    def test_full_call_has_all_keys(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "encoder_prompt" in result
        assert "decoder_input_ids" in result


class TestText2TextPromptSlotBatchProcessing:
    """Test prompt tokenization via the encoder/decoder prompt slots directly."""

    def test_encoder_slot_tokenizes_texts(self, tokenizer):
        """encoder_prompt_slot.processor.process_batch should tokenize strings."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        prompts = ["translate:", "en:"]
        ids, mask = processor.encoder_prompt_slot.processor.process_batch(prompts)
        assert isinstance(ids, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert ids.shape[0] == 2
        assert mask.shape[0] == 2

    def test_decoder_slot_tokenizes_texts(self, tokenizer):
        """decoder_prompt_slot.processor.process_batch should tokenize strings."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        prompts = ["de:", "fr:"]
        ids, mask = processor.decoder_prompt_slot.processor.process_batch(prompts)
        assert isinstance(ids, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert ids.shape[0] == 2
        assert ids.shape == mask.shape
