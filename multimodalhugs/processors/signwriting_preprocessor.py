from typing import List, Dict, Any, Optional, Callable
import logging
import torch

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.image_utils import PILImageResampling  # If used in 'frame_preprocessor'
from transformers.processing_utils import ProcessorMixin

from multimodalhugs.data import (
    pad_and_create_mask,
    center_image_on_white_background,
)

logger = logging.getLogger(__name__)


class SignwritingPreprocessor(ProcessorMixin):  # FeatureExtractionMixin
    attributes = ["frame_preprocessor", "lang_tokenizer", "tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    frame_preprocessor_class = "CLIPImageProcessor"
    tokenizer_class = "M2M100Tokenizer"
    lang_tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        frame_preprocessor: Optional[Callable] = None,
        lang_tokenizer: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        width: int = 224,
        height: int = 224,
        channels: int = 3,
        invert_frame: bool = True,
        dataset_mean: Optional[List[float]] = None,
        dataset_std: Optional[List[float]] = None,
        **kwargs,
    ):

        self.width = width
        self.height = height
        self.channels = channels
        self.invert_frame = invert_frame
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        
        super().__init__(frame_preprocessor=frame_preprocessor, lang_tokenizer=lang_tokenizer, tokenizer=tokenizer, **kwargs)

    def get_langtok(self, langtok, tokenizer):
        langtok_idx = None
        if tokenizer is not None:
            langtok_idx = tokenizer.convert_tokens_to_ids(langtok)
        return langtok_idx

    def _ascii_to_tensor(self, sign):
        """
        Converts a sequence of ascii signwritting symbols into a sequence of typo tensor images with shape [N_frames, C, W, H].
        """
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


    def __call__(
        self,
        batch: List[Dict[str, Any]],
        **kwargs,
    ) -> BatchFeature:

        tensor_secuences = [self._ascii_to_tensor(ascii_secuence["source"]) for ascii_secuence in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_secuences)

        src_langtoks = torch.stack(
            [torch.LongTensor([self.get_langtok(f"__{sample['src_lang']}__", self.lang_tokenizer)]) for sample in batch]
        )

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
        
        return BatchFeature({
            "input_frames": padded_inputs,                         # torch.Size([batch_size, n_frames, n_channes, W, H])
            "attention_mask": padded_input_masks,                  # torch.Size([batch_size, n_frames]) 0 indicates padding elements
            "src_langtoks": src_langtoks,                          # torch.Size([batch_size, 1])
            "decoder_input_ids": decoder_input_ids,                # torch.Size([batch_size, n_tokens])
            "labels": tgt_tensor,                                  # torch.Size([batch_size, n_tgt_tokens])
            "decoder_attention_mask": decoder_attention_mask,      # torch.Size([batch_size, n_tokens]) 0 indicates padding elements   
        })