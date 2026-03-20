"""Tests for SignwritingProcessor."""

import torch
from unittest.mock import patch, MagicMock
from transformers.feature_extraction_utils import BatchFeature

from multimodalhugs.processors.legacy.signwriting_preprocessor import SignwritingProcessor
from tests.test_data.conftest import SIGNWRITING_STRINGS

_MODALITY_MOCK_PATH = (
    "multimodalhugs.processors.signwriting_modality_processor.AutoProcessor.from_pretrained"
)


def _make_mock_preprocessor():
    """Create a mock AutoProcessor that returns fake pixel_values."""
    mock = MagicMock()
    mock.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    return mock


def _modality_proc(processor):
    """Return the underlying SignwritingModalityProcessor from the wrapper."""
    return processor.encoder_slots[0].processor


class TestAsciiToTensor:
    @patch(_MODALITY_MOCK_PATH)
    def test_converts_fsw_to_tensor(self, mock_from_pretrained, tokenizer):
        mock_proc = _make_mock_preprocessor()
        mock_from_pretrained.return_value = mock_proc
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        tensor = _modality_proc(processor)._ascii_to_tensor(SIGNWRITING_STRINGS[0])
        assert isinstance(tensor, torch.Tensor)
        # Should have shape [N_symbols, C, W, H]
        assert tensor.ndim == 4
        assert tensor.shape[1] == 3  # channels
        assert tensor.shape[2] == 224
        assert tensor.shape[3] == 224

    @patch(_MODALITY_MOCK_PATH)
    def test_tensor_passthrough(self, mock_from_pretrained, tokenizer):
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
        )
        t = torch.randn(5, 3, 224, 224)
        result = _modality_proc(processor).process_sample(t)
        assert torch.equal(result, t)

    @patch(_MODALITY_MOCK_PATH)
    def test_different_fsw_strings_produce_tensors(
        self, mock_from_pretrained, tokenizer
    ):
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        for fsw in SIGNWRITING_STRINGS:
            tensor = _modality_proc(processor)._ascii_to_tensor(fsw)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 4


class TestSignwritingObtainMultimodalInputAndMasks:
    @patch(_MODALITY_MOCK_PATH)
    def test_returns_input_frames_and_mask(self, mock_from_pretrained, tokenizer):
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        batch = [
            {
                "signal": SIGNWRITING_STRINGS[0],
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "test",
            },
        ]
        result = processor(batch=batch)
        assert "input_frames" in result
        assert "attention_mask" in result

    @patch(_MODALITY_MOCK_PATH)
    def test_padding_different_lengths(self, mock_from_pretrained, tokenizer):
        """Different-length FSW strings should be padded."""
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        # Two FSW strings; after normalisation both produce the same number of tokens
        batch = [
            {
                "signal": SIGNWRITING_STRINGS[0],
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "a",
            },
            {
                "signal": SIGNWRITING_STRINGS[1],
                "encoder_prompt": "",
                "decoder_prompt": "",
                "output": "b",
            },
        ]
        result = processor(batch=batch)
        assert result["input_frames"].shape[0] == 2  # batch size
        assert result["attention_mask"].shape[0] == 2


class TestSignwritingTransformGetItemsOutput:
    @patch(_MODALITY_MOCK_PATH)
    def test_converts_signals_to_tensors(self, mock_from_pretrained, tokenizer):
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        batch = {
            "signal": [SIGNWRITING_STRINGS[0]],
        }
        result = processor._transform_get_items_output(batch)
        assert isinstance(result["signal"][0], torch.Tensor)


class TestSignwritingProcessorCall:
    """Full __call__() tests — the path exercised by DataCollatorMultimodalSeq2Seq."""

    EXPECTED_KEYS = {
        "input_frames",
        "attention_mask",
        "encoder_prompt",
        "encoder_prompt_length_padding_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    }

    @patch(_MODALITY_MOCK_PATH)
    def test_returns_batch_feature(self, mock_from_pretrained, tokenizer, signwriting_batch_samples):
        """__call__ should return a BatchFeature (HF-compatible mapping)."""
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        result = processor(batch=signwriting_batch_samples)
        assert isinstance(result, BatchFeature)

    @patch(_MODALITY_MOCK_PATH)
    def test_has_all_expected_keys(self, mock_from_pretrained, tokenizer, signwriting_batch_samples):
        """Output must contain all keys consumed by the model forward()."""
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        result = processor(batch=signwriting_batch_samples)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    @patch(_MODALITY_MOCK_PATH)
    def test_batch_dimensions_consistent(self, mock_from_pretrained, tokenizer, signwriting_batch_samples):
        """Every output tensor must have the same leading batch dimension."""
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
            width=224,
            height=224,
            channels=3,
        )
        result = processor(batch=signwriting_batch_samples)
        batch_size = len(signwriting_batch_samples)
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                assert val.shape[0] == batch_size, (
                    f"Key '{key}' has batch dim {val.shape[0]}, expected {batch_size}"
                )
