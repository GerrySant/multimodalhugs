import os
import cv2
import torch
import pyarrow
import numpy as np
import logging

from typing import List, Any, Optional

from multimodalhugs.data import pad_and_create_mask, get_images
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor

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
            tensor_secuences = []
            for sample in batch:
                src_text_or_path = sample["source"]
                if os.path.exists(src_text_or_path):
                    # Determine the file extension
                    _, ext = os.path.splitext(src_text_or_path)
                    ext = ext.lower()

                    if ext == ".npy":
                        # Load npy format
                        image = np.load(src_text_or_path)
                    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                        # Load common image formats
                        image = cv2.imread(src_text_or_path, cv2.IMREAD_UNCHANGED)
                    else:
                        raise ValueError(f"Unsupported file format: {ext}")

                    if image is not None:
                        # Perform normalization if required
                        if self.normalize_image:
                            image = (image - self.mean) / self.std
                        tensor_secuences.append(torch.from_numpy(image))
                    else:
                        raise ValueError(f"Failed to read image from path: {src_text_or_path}")
                else:
                    # Treat it as text
                    tensor_secuences.append(torch.from_numpy(
                        get_images(
                            src_text=src_text_or_path,
                            font_path=self.font_path,
                            width=self.width,
                            height=self.height,
                            normalize_image=self.normalize_image,
                            mean=self.mean,
                            std=self.std,
                        )))

        elif isinstance(batch[0]["source"], numpy.ndarray):
            tensor_secuences = [torch.from_numpy(sample["source"]) for sample in batch]
        else:
            raise TypeError("Unsupported type for 'source'.")
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_secuences)
        return {
            "input_frames": padded_inputs,                         # torch.Size([batch_size, n_frames, n_channes, W, H])
            "attention_mask": padded_input_masks                   # torch.Size([batch_size, n_frames]) 0 indicates padding elements
        }, kwargs