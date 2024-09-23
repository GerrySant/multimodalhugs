import torch
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


class Pose2TextTranslationPreprocessor(ProcessorMixin):  # FeatureExtractionMixin
    attributes = ["lang_tokenizer", "tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"
    lang_tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        lang_tokenizer: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        reduce_holistic_poses: bool = True,
        obtainables_list: Optional[List[str]] = None,
        **kwargs,
    ):
        self.reduce_holistic_poses = reduce_holistic_poses
        self.obtainables_list = obtainables_list
        super().__init__(lang_tokenizer=lang_tokenizer, tokenizer=tokenizer, **kwargs)
        
    def get_obtainables(self):
        if self.obtainables_list is not None:
            return [getattr(self, method_name) for method_name in self.obtainables_list]
        else:
            return [getattr(self, method_name) for method_name in dir(self) if method_name.startswith('_obtain_')]

    def get_langtok(self, langtok, tokenizer):
        langtok_idx = None
        if tokenizer is not None:
            langtok_idx = tokenizer.convert_tokens_to_ids(langtok)
        return langtok_idx
            
    def _pose_file_to_tensor(self, pose_file: Union[str, Path]):
        pose_file = open(pose_file, "rb").read()
        pose = Pose.read(pose_file) # [t, people, d, xyz]
    
        P1 = ("POSE_LANDMARKS", "RIGHT_SHOULDER") if pose.header.components[0].name == "POSE_LANDMARKS" else ("BODY_135", "RShoulder")
        P2 = ("POSE_LANDMARKS", "LEFT_SHOULDER") if pose.header.components[0].name == "POSE_LANDMARKS" else ("BODY_135", "LShoulder")
    
        pose_hide_legs(pose)
    
        if self.reduce_holistic_poses:
            pose = reduce_holistic(pose) # [t, people, d', xyz]
        
        pose = pose.normalize(pose.header.normalization_info(p1=P1, p2=P2))
        pose = pose.torch().body.data.squeeze(1) # [t, d', xyz]
        pose = pose.reshape(shape=(pose.shape[0], -1))
        return pose.zero_filled()
        
    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_secuences = [self._pose_file_to_tensor(sample["source"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_secuences)
        return {
            "inputs_embeds": padded_inputs,                         # torch.Size([batch_size, n_frames, n_channes, W, H])
            "attention_mask": padded_input_masks                   # torch.Size([batch_size, n_frames]) 0 indicates padding elements
        }, kwargs

    def _obtain_src_langtoks(self, batch, **kwargs):
        src_langtoks = torch.stack(
            [torch.LongTensor([self.get_langtok(f"__{sample['src_lang']}__", self.lang_tokenizer)]) for sample in batch]
        ) if self.lang_tokenizer is not None else None
        return {
            "src_langtoks": src_langtoks,                          # torch.Size([batch_size, 1])
        }, kwargs

    def _obtain_others(self, batch, **kwargs):
        return self._obtain_src_langtoks(batch, **kwargs)
        
    def _obtain_text_inputs_and_labels(self, batch, **kwargs):
        tgt_langtoks = torch.stack(
            [torch.LongTensor([self.get_langtok(f"__{sample['tgt_lang']}__", self.tokenizer)]) for sample in batch]
        )

        tokenization = self.tokenizer(
            text=[sample["tgt_sentence"] for sample in batch],
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        
        tgt_tensor = tokenization['input_ids']                              # ['<token_a>', '<token_b>', '<token_c>', '</s>']
        tgt_tensor = torch.cat((tgt_langtoks, tgt_tensor[:, 1:]), dim=1)    # ['<tgt_lang>', '<token_a>', '<token_b>', '<token_c>', '</s>']

        decoder_attention_mask = tokenization['attention_mask']
        decoder_attention_mask = torch.cat((torch.full((decoder_attention_mask.size(0), 1), 1), decoder_attention_mask), dim=1)
        decoder_attention_mask = decoder_attention_mask[..., :-1].contiguous()
        
        # Prepare final output for model consumption
        decoder_input_ids = torch.full((tgt_tensor.size(0), 1), self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token))
        decoder_input_ids = torch.cat((decoder_input_ids, tgt_tensor), dim=1)   # ['</s>', '<tgt_lang>', '<token_a>', '<token_b>', '<token_c>', '</s>']
        decoder_input_ids = decoder_input_ids[..., :-1].contiguous()            # ['</s>', '<tgt_lang>', '<token_a>', '<token_b>', '<token_c>']
        return {
            "decoder_input_ids": decoder_input_ids,                # torch.Size([batch_size, n_tokens])
            "labels": tgt_tensor,                                  # torch.Size([batch_size, n_tgt_tokens])
            "decoder_attention_mask": decoder_attention_mask,      # torch.Size([batch_size, n_tokens]) 0 indicates padding elements   
        }, kwargs
        
    def __call__(
        self,
        batch: List[Dict[str, Any]],
        **kwargs,
    ) -> BatchFeature:
        
        batch_dict = {}
        for obtain_method in self.get_obtainables():
            obgained_dict, kwargs = obtain_method(batch, **kwargs)
            batch_dict.update(obgained_dict)
            
        return BatchFeature(batch_dict)          
