import os
import torch
import random
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
        target_lang_on_source: bool = False,
        task_prefixes: list = [],
        **kwargs,
    ):
        obtainables_list = kwargs.pop('obtainables_list', None)
        self.target_lang_on_source = target_lang_on_source
        self.obtainables_list = obtainables_list
        self.task_prefixes = task_prefixes
    
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

    def create_prompt_strings(self, sample):
        if len(self.task_prefixes) > 0:
            batch_keys = ["task"]
            task = sample.get('task', None)
            sample["task"] = random.choice(self.task_prefixes) if task is None else task
        else:
            batch_keys = []
            sample["task"] = None
        batch_keys.append("src_lang")
        if self.target_lang_on_source:
            batch_keys.append("tgt_lang")
        prompt = " ".join(f"__{sample[key]}__" for key in batch_keys if key in sample and sample[key] is not None)
        return prompt if prompt else None

    def _obtain_whatever(self, batch, **kwargs):
        raise NotImplementedError("_obtain_<whatever> methods must be implemented by the child class.")
        
    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        raise NotImplementedError("_obtain_multimodal_input_and_masks method must be implemented by the child class.")

    def _obtain_src_prompt(self, batch, **kwargs):
        src_prompt_list = [self.create_prompt_strings(sample) for sample in batch]
        src_prompt = torch.stack([torch.LongTensor(
            self.tokenizer.encode(
                text=sample, 
                add_special_tokens=False, 
                padding=False, 
                truncation=None, 
                max_length=None, 
                stride=0, 
                return_tensors=None,
            )) for sample in src_prompt_list]) if self.tokenizer is not None else None
        return {
            "src_prompt": src_prompt,                          # torch.Size([batch_size, 1])
        }, kwargs

    def _obtain_others(self, batch, **kwargs):
        return self._obtain_src_prompt(batch, **kwargs)
        
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
