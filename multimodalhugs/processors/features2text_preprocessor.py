import torch
import logging
import numpy as np

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

from multimodalhugs.data import (
    pad_and_create_mask,
)
from transformers.processing_utils import ProcessorMixin
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor

logger = logging.getLogger(__name__)


class Features2TextTranslationProcessor(MultimodalSecuence2TextTranslationProcessor):  # FeatureExtractionMixin
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _features_file_to_tensor(
        self, 
        features_file: Union[str, Path], 
    ) -> torch.Tensor:
        """
        Converts features from .npy file to a tensor representation.
        
        Args:
            features_file (Union[str, Path]): Path to the features file.
        Returns:
            torch.Tensor: Tensor representation of the features file.
        """
        return torch.from_numpy(np.load(features_file))

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_secuences = [self._features_file_to_tensor(sample["source"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_secuences)
        return {
            "inputs_embeds": padded_inputs,                         # torch.Size([batch_size, n_frames, *])
            "attention_mask": padded_input_masks                   # torch.Size([batch_size, n_frames]) 0 indicates padding elements
        }, kwargs

