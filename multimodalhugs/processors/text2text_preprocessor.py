import os
import cv2
import torch
import pyarrow
import logging
import numpy as np

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
    get_images
)
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs

logger = logging.getLogger(__name__)


class Text2TextTranslationProcessor(MultimodalSequence2SequenceProcessor):  # FeatureExtractionMixin
    name = "text2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        # Tokenize the prompts with padding
        tokenized_output = self.tokenizer(
            [sample['signal'] for sample in batch],
            add_special_tokens=False,
            padding=True,  # Automatically add padding
            truncation=False,  # Do not truncate sequences
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Obtain prompt tensors and the mask
        padded_signal = tokenized_output["input_ids"]
        signal_length_padding_mask = tokenized_output["attention_mask"]

        return {
            "input_ids": padded_signal,                         # torch.Size([batch_size, n_frames)
            "attention_mask": signal_length_padding_mask         # torch.Size([batch_size, n_frames]) 0 indicates padding elements
        }, kwargs