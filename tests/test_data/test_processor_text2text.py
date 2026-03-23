"""Tests for Text2TextTranslationProcessor."""

import torch
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.legacy.text2text_preprocessor import (
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
        # Both should have same sequence length (padded)
        assert result["input_ids"].shape[1] == result["attention_mask"].shape[1]


class TestText2TextObtainPrompts:
    def test_encoder_prompt(self, tokenizer, text_batch_samples):
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        result = processor(batch=text_batch_samples)
        assert "encoder_prompt" in result
        assert "encoder_prompt_length_padding_mask" in result
        assert result["encoder_prompt"].shape[0] == len(text_batch_samples)

    def test_decoder_prompt(self, tokenizer, text_batch_samples):
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
        # Should have input_ids, attention_mask from multimodal
        # encoder_prompt, encoder_prompt_length_padding_mask from encoder prompt
        # decoder_input_ids, decoder_attention_mask from decoder prompt
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "encoder_prompt" in result
        assert "decoder_input_ids" in result


class TestText2TextProcessPrompts:
    def test_encoder_prompt_processor_tokenizes(self, tokenizer):
        """The encoder-prompt slot's processor should tokenize text."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        # slots[2] is the encoder_prompt slot
        enc_proc = processor.slots[2].processor
        prompts = ["translate:", "en:"]
        padded, mask = enc_proc.process_batch(prompts)
        assert isinstance(padded, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert padded.shape[0] == 2
        assert mask.shape[0] == 2

    def test_decoder_prompt_processor_tokenizes(self, tokenizer):
        """The decoder-prompt slot's processor should tokenize text."""
        processor = Text2TextTranslationProcessor(tokenizer=tokenizer)
        # slots[3] is the decoder_prompt slot
        dec_proc = processor.slots[3].processor
        prompts = ["de:", "fr:"]
        padded, mask = dec_proc.process_batch(prompts)
        assert isinstance(padded, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert padded.shape[0] == 2
        assert mask.shape[0] == 2
