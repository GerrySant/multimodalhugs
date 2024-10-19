import torch
import pyarrow
import logging

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
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs

logger = logging.getLogger(__name__)


class Image2TextTranslationProcessor(MultimodalSecuence2TextTranslationProcessor):  # FeatureExtractionMixin
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        font_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        normalize_image: bool = True,
        mean: Optional[List] = None,
        std: Optional[List] = None,
        **kwargs,
    ):
        self.font_path = font_path
        self.width = width
        self.height = height
        self.normalize_image = normalize_image
        self.mean = mean
        self.std = std
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        if isinstance(batch[0]["source"], pyarrow.lib.StringScalar):
            tensor_secuences = [torch.from_numpy(
                get_images(
                    src_text=sample["source"].as_py(),
                    font_path=self.font_path,
                    width=self.width,
                    height=self.height,
                    normalize_image=self.normalize_image,
                    mean=self.mean,
                    std=self.std,
                    )) for sample in batch]

        elif isinstance(batch[0]["source"], str):
            tensor_secuences = [torch.from_numpy(
                get_images(
                    src_text=sample["source"],
                    font_path=self.font_path,
                    width=self.width,
                    height=self.height,
                    normalize_image=self.normalize_image,
                    mean=self.mean,
                    std=self.std,
                    )) for sample in batch]

        elif isinstance(batch[0]["source"], numpy.ndarray):
            tensor_secuences = [torch.from_numpy(sample["source"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_secuences)
        return {
            "input_frames": padded_inputs,                         # torch.Size([batch_size, n_frames, n_channes, W, H])
            "attention_mask": padded_input_masks                   # torch.Size([batch_size, n_frames]) 0 indicates padding elements
        }, kwargs

