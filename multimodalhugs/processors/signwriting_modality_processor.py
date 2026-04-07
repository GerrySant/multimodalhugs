import logging
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import ImageOps
from transformers import AutoProcessor

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image

from multimodalhugs.data import pad_and_create_mask, center_image_on_white_background
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput

logger = logging.getLogger(__name__)


class SignwritingModalityProcessor(ModalityProcessor):
    """
    Converts SignWriting ASCII strings into sequences of image tensors.

    process_sample — normalises an ASCII SignWriting string, renders each
                     symbol as an image via ``signwriting_to_image``, resizes
                     and optionally inverts, then applies ``custom_preprocessor``
                     to produce a [N_signs, C, H, W] float tensor.
    process_batch  — pads a list of [N_i, C, H, W] tensors to
                     [B, N_max, C, H, W] and returns a [B, N_max] mask.
    """

    def __init__(
        self,
        custom_preprocessor_path: Optional[str] = None,
        width: int = 224,
        height: int = 224,
        channels: int = 3,
        invert_frame: bool = True,
    ):
        """
        Args:
            custom_preprocessor_path: HuggingFace model ID or local path to an
                image preprocessor (e.g. ``"openai/clip-vit-base-patch32"``).
                Applied to each rendered sign image to produce the final tensor.
                Required — raises ``ValueError`` immediately if None.
            width: Target canvas width in pixels for each rendered sign symbol.
                Default: 224.
            height: Target canvas height in pixels for each rendered sign symbol.
                Default: 224.
            channels: Number of colour channels in the output tensor (e.g. 3
                for RGB). Default: 3.
            invert_frame: If True, inverts pixel values of each rendered sign
                image (black symbols on white → white symbols on black).
                Default: True.
        """
        if custom_preprocessor_path is None:
            raise ValueError(
                "SignwritingModalityProcessor requires a 'custom_preprocessor_path' "
                "(a HuggingFace model ID or local path to an image preprocessor such as "
                "CLIPImageProcessor). Pass it as a constructor argument or set it in the "
                "processor_kwargs of the slot config."
            )
        self.custom_preprocessor_path = custom_preprocessor_path
        self.width = width
        self.height = height
        self.channels = channels
        self.invert_frame = invert_frame
        self.custom_preprocessor = AutoProcessor.from_pretrained(custom_preprocessor_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ascii_to_tensor(self, sign: str) -> torch.Tensor:
        """
        Convert an ASCII SignWriting string to a sequence of image tensors.

        Normalises the input string, splits it into individual sign symbols,
        renders each symbol as an image, optionally inverts colours, and
        applies the ``custom_preprocessor`` to each frame.

        Args:
            sign: An ASCII SignWriting string (FSW format), potentially
                containing multiple sign symbols.

        Returns:
            Float tensor of shape [N_signs, C, H, W] where N_signs is the
            number of individual sign symbols in the input string.
        """
        sign_arrays = []
        for ascii_sign in normalize_signwriting(sign).split():
            _sign = signwriting_to_image(ascii_sign, trust_box=False)
            _sign = center_image_on_white_background(
                _sign, target_width=self.width, target_height=self.height
            )
            if self.invert_frame:
                _sign = ImageOps.invert(_sign)
            _sign = self.custom_preprocessor(images=_sign, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            sign_arrays.append(_sign)
        return torch.stack(sign_arrays, dim=0)

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert a single SignWriting sample to a tensor. Called at dataset-transform time.

        Args:
            values: One of:
                - str — ASCII SignWriting string (FSW format); rendered to a
                  sequence of sign images via ``_ascii_to_tensor``.
                - torch.Tensor — returned unchanged (already preprocessed).

        Returns:
            Float tensor of shape [N_signs, C, H, W].
        """
        if isinstance(values, torch.Tensor):
            return values
        return self._ascii_to_tensor(values)

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """
        Pad a batch of SignWriting tensors to a common length. Called at collation time.

        Args:
            samples: List of B tensors, each of shape [N_i, C, H, W], as
                returned by ``process_sample``.

        Returns:
            ProcessBatchOutput where:
                - data: Float tensor of shape [B, N_max, C, H, W], zero-padded.
                - mask: Bool tensor of shape [B, N_max], True for valid signs.
        """
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
