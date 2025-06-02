import os
import torch
import logging
import psutil

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.image_utils import PILImageResampling  # If used in 'frame_preprocessor'
from transformers.processing_utils import ProcessorMixin

from multimodalhugs.data import (
    pad_and_create_mask,
    center_image_on_white_background,
)
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs, pose_normalization_info

logger = logging.getLogger(__name__)


class Pose2TextTranslationProcessor(MultimodalSecuence2TextTranslationProcessor):  # FeatureExtractionMixin
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        reduce_holistic_poses: bool = True,
        **kwargs,
    ):
        self.reduce_holistic_poses = reduce_holistic_poses
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _pose_file_to_tensor(
        self, 
        pose_file: Union[str, Path, torch.Tensor], 
        signal_start: int = 0, 
        signal_end: int = 0
    ) -> torch.Tensor:
        """
        Converts a pose file to a tensor representation.
        
        Args:
            pose_file (Union[str, Path]): Path to the pose file.
            signal_start (int): Starting time (ms) (default is 0).
            signal_end (int): Ending time (ms) (default is 0).

        Returns:
            torch.Tensor: Tensor representation of the pose file.
        """
        if isinstance(pose_file, torch.Tensor):
            # Case in which the transformation has already been performed during the dataset._get_items_()
            return pose_file
        
        with open(pose_file, "rb") as pose_file:
            pose = Pose.read(pose_file, start_time=signal_start or None, end_time=signal_end or None) 
        
        pose_hide_legs(pose)
    
        if self.reduce_holistic_poses:
            pose = reduce_holistic(pose)  # [t, people, d', xyz]
        
        pose = pose.normalize()
        tensor = pose.torch().body.data.zero_filled()
        return tensor.contiguous().view(tensor.size(0), -1)

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_sequences = [self._pose_file_to_tensor(sample["signal"], sample["signal_start"], sample["signal_end"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_sequences)
        return {
            "inputs_embeds": padded_inputs,
            "attention_mask": padded_input_masks
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
        tensor_signals = [self._pose_file_to_tensor(s, start, end) 
                          for s, start, end in zip(batch["signal"], batch["signal_start"], batch["signal_end"])]
        batch["signal"] = tensor_signals
        return batch
