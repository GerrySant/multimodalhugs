import torch
import logging

from pathlib import Path
from PIL import Image, ImageOps
from typing import List, Dict, Any, Optional, Callable, Union

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers import AutoProcessor
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor

from multimodalhugs.data import (
    pad_and_create_mask,
    center_image_on_white_background,
)

logger = logging.getLogger(__name__)


class SignwritingProcessor(MultimodalSequence2SequenceProcessor):  # FeatureExtractionMixin
    name = "signwritting2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        custom_preprocessor_path: Optional[str] = None,
        width: int = 224, #
        height: int = 224, #
        channels: int = 3, #
        invert_frame: bool = True, #
        **kwargs,
    ):
        """
        Initializes the SignwritingProcessor for processing signwriting frames and converting them into text-compatible inputs.

        Args:
            custom_preprocessor_path (Optional[str], optional): Path of a custom imaage procesor used to preprocess the image frames.
                Typically an instance of a Hugging Face image processor (e.g., CLIPImageProcessor).
            tokenizer (Optional[Any], optional): A tokenizer object used to tokenize the text output. Usually loaded via Hugging Face's AutoTokenizer.
            width (int, optional): Target width (in pixels) for images/frames after preprocessing. Defaults to 224.
            height (int, optional): Target height (in pixels) for images/frames after preprocessing. Defaults to 224.
            channels (int, optional): Number of color channels in the images/frames (e.g., 3 for RGB). Defaults to 3.
            invert_frame (bool, optional): If True, inverts pixel values for preprocessing. Useful if input images are white-on-black. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent class (`MultimodalSequence2SequenceProcessor`), such as `max_seq_length`, `padding`, or modality-specific parameters.
        """

        self.width = width #
        self.height = height #
        self.channels = channels #
        self.invert_frame = invert_frame #
        self.custom_preprocessor_path = custom_preprocessor_path
        self.custom_preprocessor = AutoProcessor.from_pretrained(self.custom_preprocessor_path) if self.custom_preprocessor_path is not None else None
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _ascii_to_tensor(self, sign):
        """
        Converts a sequence of ascii signwritting symbols into a sequence of typo tensor images with shape [N_frames, C, W, H].
        """
        if isinstance(sign, torch.Tensor):
            # Case in which the transformation has already been performed during the dataset._get_items_()
            return sign
        
        sign_arrays = []
        for ascii_sign in normalize_signwriting(sign).split():
            if ascii_sign == "M511x510S27034490x490":
                ascii_sign = "M511x510S2c734490x490"
            if ascii_sign == "M510x518S21005490x483":
                ascii_sign = "M510x518S2c105490x483"
            _sign = signwriting_to_image(ascii_sign, trust_box=False)
            _sign = center_image_on_white_background(_sign, target_width=self.width, target_height=self.height)
            if self.invert_frame:
                _sign = ImageOps.invert(_sign)
            _sign = self.custom_preprocessor(images=_sign, return_tensors="pt")['pixel_values'].squeeze(0) 

            try:
                assert isinstance(_sign, torch.Tensor), "sign must be a torch tensor"
            except AssertionError as error:
                logger.error(error)
            try:
                assert list(_sign.shape) == [self.channels, self.width, self.height], f"Expected a tensor of shape of ({self.channels}, {self.width}, {self.height}), but a tensor of shape {_sign.shape} was given)"
            except AssertionError as error:
                logger.error(error)
            sign_arrays.append(_sign)
        signs_concatenated = torch.stack(sign_arrays, dim=0)
        return signs_concatenated

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_secuences = [self._ascii_to_tensor(sample["signal"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_secuences)
        return {
            "input_frames": padded_inputs,                         # torch.Size([batch_size, n_frames, n_channes, W, H])
            "attention_mask": padded_input_masks                   # torch.Size([batch_size, n_frames]) 0 indicates padding elements
        }, kwargs     

    def _transform_get_items_output(self, batch):
        """
        Returns a transformation function applied at the dataset level during iteration.

        This method defines a transformation that is applied to each batch **within the dataset iterator**, 
        typically by using `datasets.Dataset.with_transform()`. As a result, the transformation is executed 
        at runtime during `__getitem__()` or `__getitems__()`, which allows it to benefit from prefetching 
        and parallel data loading when using multiple DataLoader workers.

        Unlike the `_obtain_*` methods, which are also executed on-the-fly but within the **processor call 
        (typically inside the DataCollator)**, this transformation occurs **prior to batching and collation**. 
        It is therefore ideal for operations that are expensive and can be parallelized at the sample or batch 
        level, such as decoding signals, loading external files, or converting inputs to intermediate formats.

        Use this method to preprocess inputs early in the pipeline while maintaining a modular design that 
        separates dataset-level and collator-level responsibilities.

        Args:
            batch (Dict[str, List[Any]]): A dictionary representing a batch of dataset examples (not yet collated).

        Returns:
            Dict[str, List[Any]]: The transformed batch, with updated or added fields ready for collation.
        """
        tensor_signals = [self._ascii_to_tensor(batch["signal"][i]) for i in range(len(batch["signal"]))]
        batch["signal"] = tensor_signals
        return batch