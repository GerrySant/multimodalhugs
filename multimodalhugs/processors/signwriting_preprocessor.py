import torch
import logging

from pathlib import Path
from PIL import Image, ImageOps
from typing import List, Dict, Any, Optional, Callable, Union

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.image_utils import PILImageResampling  # If used in 'frame_preprocessor'
from transformers.processing_utils import ProcessorMixin
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor

from multimodalhugs.data import (
    pad_and_create_mask,
    center_image_on_white_background,
)

logger = logging.getLogger(__name__)



class SignwritingProcessor(MultimodalSequence2SequenceProcessor):  # FeatureExtractionMixin
    name = "signwritting2text_processor"
    attributes = ["frame_preprocessor", "tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    frame_preprocessor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        frame_preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        width: int = 224, #
        height: int = 224, #
        channels: int = 3, #
        invert_frame: bool = True, #
        dataset_mean: Optional[List[float]] = None, #
        dataset_std: Optional[List[float]] = None, #
        **kwargs,
    ):

        self.width = width #
        self.height = height #
        self.channels = channels #
        self.invert_frame = invert_frame #
        self.dataset_mean = dataset_mean #
        self.dataset_std = dataset_std #
        
        super().__init__(frame_preprocessor=frame_preprocessor, tokenizer=tokenizer, **kwargs)

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

            _sign = self.frame_preprocessor(_sign, return_tensors="pt")['pixel_values'].squeeze(0) 

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