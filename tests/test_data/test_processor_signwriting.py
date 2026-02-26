"""Tests for SignwritingProcessor."""

import torch
from unittest.mock import patch, MagicMock

from multimodalhugs.processors.signwriting_preprocessor import SignwritingProcessor
from tests.test_data.conftest import SIGNWRITING_STRINGS


def _make_mock_preprocessor():
    """Create a mock AutoProcessor that returns fake pixel_values."""
    mock = MagicMock()
    mock.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    return mock


class TestAsciiToTensor:
    @patch(
        "multimodalhugs.processors.signwriting_preprocessor.AutoProcessor.from_pretrained"
    )
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
        tensor = processor._ascii_to_tensor(SIGNWRITING_STRINGS[0])
        assert isinstance(tensor, torch.Tensor)
        # Should have shape [N_symbols, C, W, H]
        assert tensor.ndim == 4
        assert tensor.shape[1] == 3  # channels
        assert tensor.shape[2] == 224
        assert tensor.shape[3] == 224

    @patch(
        "multimodalhugs.processors.signwriting_preprocessor.AutoProcessor.from_pretrained"
    )
    def test_tensor_passthrough(self, mock_from_pretrained, tokenizer):
        mock_from_pretrained.return_value = _make_mock_preprocessor()
        processor = SignwritingProcessor(
            tokenizer=tokenizer,
            custom_preprocessor_path="mock/path",
        )
        t = torch.randn(5, 3, 224, 224)
        result = processor._ascii_to_tensor(t)
        assert torch.equal(result, t)

    @patch(
        "multimodalhugs.processors.signwriting_preprocessor.AutoProcessor.from_pretrained"
    )
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
            tensor = processor._ascii_to_tensor(fsw)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 4


class TestSignwritingObtainMultimodalInputAndMasks:
    @patch(
        "multimodalhugs.processors.signwriting_preprocessor.AutoProcessor.from_pretrained"
    )
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
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
        assert "input_frames" in result
        assert "attention_mask" in result

    @patch(
        "multimodalhugs.processors.signwriting_preprocessor.AutoProcessor.from_pretrained"
    )
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
        # First FSW has 2 symbols, second has 2, third has 2
        # (simple strings with 2 space-separated tokens after normalize)
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
        result, _ = processor._obtain_multimodal_input_and_masks(batch)
        assert result["input_frames"].shape[0] == 2  # batch size
        assert result["attention_mask"].shape[0] == 2


class TestSignwritingTransformGetItemsOutput:
    @patch(
        "multimodalhugs.processors.signwriting_preprocessor.AutoProcessor.from_pretrained"
    )
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
