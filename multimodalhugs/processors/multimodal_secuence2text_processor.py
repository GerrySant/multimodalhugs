import os
import torch
import random
import logging

from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
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
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs


logger = logging.getLogger(__name__)


class MultimodalSecuence2TextTranslationProcessor(ProcessorMixin):  # FeatureExtractionMixin
    
    attributes = ["frame_preprocessor", "tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    frame_preprocessor_class = "BaseImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs: List[str] = ["obtainables_list",]

    def __init__(
        self,
        frame_preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        obtainables_list = kwargs.pop('obtainables_list', None)
        self.obtainables_list = obtainables_list
    
        if frame_preprocessor is None:
            super().__init__(
                tokenizer=tokenizer, 
                **kwargs)
        else:
            super().__init__(
                frame_preprocessor=frame_preprocessor, 
                tokenizer=tokenizer, 
                **kwargs
            )

    def process_prompts(self, prompts):
        """
        Processes the prompts to account for different token lengths among samples.

        Args:
            prompts (list of str): List of prompts.

        Returns:
            torch.Tensor: Prompt tensors with padding.
            torch.Tensor: Mask indicating the padding positions (1 for real tokens, 0 for padding).
        """

        # Tokenize the prompts with padding
        tokenized_output = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,  # Automatically add padding
            truncation=False,  # Do not truncate sequences
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Obtain prompt tensors and the mask
        padded_prompts = tokenized_output["input_ids"]
        prompt_length_padding_mask = tokenized_output["attention_mask"]

        return padded_prompts, prompt_length_padding_mask

    def get_obtainables(self):
        if self.obtainables_list is not None:
            obtainables = [getattr(self, method_name) for method_name in self.obtainables_list]
        else:
            obtainables = [getattr(self, method_name) for method_name in dir(self) if method_name.startswith('_obtain_')]

        # Remove '_obtain_whatever' only if there are other methods in the list
        obtain_whatever_method = getattr(self, '_obtain_whatever', None)
        if obtain_whatever_method and len(obtainables) > 1:
            obtainables = [method for method in obtainables if method != obtain_whatever_method]
        
        return obtainables

    def get_langtok(self, langtok):
        langtok_idx = None
        if self.tokenizer is not None:
            langtok_idx = self.tokenizer.convert_tokens_to_ids(langtok)
        return langtok_idx

    def _obtain_whatever(self, batch, **kwargs):
        raise NotImplementedError("_obtain_<whatever> methods must be implemented by the child class.")
        
    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        raise NotImplementedError("_obtain_multimodal_input_and_masks method must be implemented by the child class.")

    def _obtain_encoder_prompt(self, batch, **kwargs):
        padded_prompts, encoder_prompt_length_padding_mask = self.process_prompts([sample['encoder_prompt'] for sample in batch])

        return {
            "encoder_prompt": padded_prompts,                                               # torch.Size([batch_size, prompt_length])
            "encoder_prompt_length_padding_mask": encoder_prompt_length_padding_mask,     # torch.Size([batch_size, prompt_length])
        }, kwargs

    def _obtain_decoder_prompt(self, batch, **kwargs):
        padded_prompts, decoder_prompt_length_padding_mask = self.process_prompts([sample['decoder_prompt'] for sample in batch])

        return {
            "decoder_input_ids": padded_prompts,                                 # torch.Size([batch_size, prompt_length])
            "decoder_attention_mask": decoder_prompt_length_padding_mask,     # torch.Size([batch_size, prompt_length])
        }, kwargs
        
    def __call__(
        self,
        batch: List[Dict[str, Any]],
        batch_dict: Optional[Dict[str, Any]] = {},
        **kwargs,
    ) -> BatchFeature:

        for obtain_method in self.get_obtainables():
            obgained_dict, kwargs = obtain_method(batch, **kwargs)
            batch_dict.update(obgained_dict)
            
        return BatchFeature(batch_dict)     
