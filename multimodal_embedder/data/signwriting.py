import os
from pathlib import Path
from typing import Any
import logging

import torch
from PIL import ImageOps
from datasets import load_dataset
from torchvision.transforms import Compose
from transformers import M2M100Tokenizer

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image
from multimodal_embedder.data import (
    MultimodalMTDataConfig,
    check_columns,
    load_tokenizer_from_vocab_file,
    pad_and_create_mask,
    center_image_on_white_background,
    contains_empty,
)
from multimodal_embedder.custom_datasets import properly_format_signbank_plus

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['source', 'target', 'tgt_lang', 'src_lang']

class SignWritingDataset(torch.utils.data.Dataset):
    def __init__(self, metafile_path, split: str, config: MultimodalMTDataConfig, preprocess_fn: Any = None):
        self.config = config
        self.dataset = load_dataset('csv', data_files=[str(metafile_path)], split=split)
        if not check_columns(self.dataset, REQUIRED_COLUMNS):
            dir_name, file_name = os.path.split(metafile_path)
            new_file_name = "corrected_" + file_name
            out_path = Path(dir_name) / new_file_name
            if not out_path.exists():
                properly_format_signbank_plus(metafile_path)
            self.dataset = load_dataset('csv', data_files=[str(out_path)], split=split)

        inicial_n_samples = len(self.dataset)
        self.dataset = self.dataset.filter(lambda sample: not contains_empty(sample))
        inicial_n_samples = inicial_n_samples - len(self.dataset)
        logger.info(f'{inicial_n_samples} samples were filtered out as they contained empty values.')
        
        self.src_tokenizer = load_tokenizer_from_vocab_file(config.src_lang_tokenizer_path)
        self.remove_unused_columns = config.remove_unused_columns
        self.tgt_tokenizer = M2M100Tokenizer.from_pretrained(config.text_tokenizer_path)
        self.preprocess = preprocess_fn
        
    def get_langtok(self, langtok, tokenizer):
        langtok_idx = None
        if tokenizer is not None:
            langtok_idx = tokenizer.convert_tokens_to_ids(langtok)
        return langtok_idx
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """
        Retrieve an item by index from the dataset and process it for model input.

        Args:
        idx (int): Index of the item in the dataset.

        Returns:
        dict: A dictionary containing processed source and target language information.
        """
        # Retrieve the dataset item
        item = self.dataset[idx]

        # Process source language information
        src_lang_token = self.get_langtok(f"__{item['src_lang']}__", self.src_tokenizer)
        sign_sequence = self.postprocess(sign=item['source'])
        _item = {"id": idx, "src_lang": src_lang_token, "source": sign_sequence}
        
        # Process target language information
        _item['tgt_langtok'] = self.get_langtok(f"__{item['tgt_lang']}__", self.tgt_tokenizer)
        _item['tgt_tensor'] = self.tgt_tokenizer(
            text=item['target'],
            return_tensors='pt', 
            padding=False, 
            truncation=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )['input_ids'].squeeze()
        _item['tgt_sentence'] = item['target']
        if self.remove_unused_columns:
            return {'input_ids':_item}
        return _item

    def collate_fn(self, batch):
        """
        Collate function to process a batch of items and prepare them for model input.

        Args:
        batch (list of dicts): The batch to process.

        Returns:
        dict: A dictionary containing collated inputs and targets for the model.
        """
        
        if len(batch) > 0:
            if len(batch[0]) == 1 and 'input_ids' in batch[0].keys():
                batch = [sample['input_ids'] for sample in batch]

        # Extract basic information from batch
        ids = [sample['id'] for sample in batch]
        input_tensors = [sample['source'] for sample in batch]

        # Padding input tensors
        padded_inputs, padded_input_masks = pad_and_create_mask(input_tensors)

        # Handle source language tokens
        if all("src_lang" in sample for sample in batch):
            src_langtoks = torch.stack([torch.LongTensor([sample["src_lang"]]) for sample in batch])
        else:
            src_langtoks = None

        collated =  {
            "input_ids": padded_inputs,             # TODO: Passing None leads to an error at ModuleUtilsMixin.floating_point_ops(self, input_dict, exclude_embeddings) from transformers
            "input_frames": padded_inputs,          # torch.Size([batch_size, n_frames, n_channes, W, H])
            "attention_mask": padded_input_masks,   # torch.Size([batch_size, n_frames]) 0 indicates padding elements
            "src_langtoks": src_langtoks,           # torch.Size([batch_size, 1])
        }

        # Process target language information
        tgt_sentences = [sample["tgt_sentence"] for sample in batch if sample["id"] in ids]
        tgt_langtoks = torch.stack([torch.LongTensor([sample["tgt_langtok"]]) for sample in batch if sample["id"] in ids])

        # Tokenization for target language
        tokenization = self.tgt_tokenizer(
            text=tgt_sentences,
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        # Adjusting target tensor and masks
        tgt_tensor = tokenization['input_ids']                              # ['<token_a>', '<token_b>', '<token_c>', '</s>']
        tgt_tensor = torch.cat((tgt_langtoks, tgt_tensor[:, 1:]), dim=1)    # ['<tgt_lang>', '<token_a>', '<token_b>', '<token_c>', '</s>']

        decoder_attention_mask = tokenization['attention_mask']
        decoder_attention_mask = torch.cat((torch.full((decoder_attention_mask.size(0), 1), 1), decoder_attention_mask), dim=1)
        decoder_attention_mask = decoder_attention_mask[..., :-1].contiguous()

        # Prepare final output for model consumption
        ntokens = torch.sum(decoder_attention_mask, dim=1)
        decoder_input_ids = torch.full((tgt_tensor.size(0), 1), self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.eos_token))
        decoder_input_ids = torch.cat((decoder_input_ids, tgt_tensor), dim=1)   # ['</s>', '<tgt_lang>', '<token_a>', '<token_b>', '<token_c>', '</s>']
        decoder_input_ids = decoder_input_ids[..., :-1].contiguous()            # ['</s>', '<tgt_lang>', '<token_a>', '<token_b>', '<token_c>']

        # Collating all necessary information for model input
        collated["decoder_input_ids"] = decoder_input_ids           # torch.Size([batch_size, n_tokens]) (before: tgt_tensor)
        collated["labels"] = tgt_tensor                             # torch.Size([batch_size, n_tgt_tokens])
        collated["decoder_attention_mask"] = decoder_attention_mask # torch.Size([batch_size, n_tokens]) 0 indicates padding elements
        collated["ntokens"] = ntokens                               # torch.Size([batch_size]
        
        return collated
        
    def postprocess(self, sign):     
        sign_arrays = []
        for ascii_sign in normalize_signwriting(sign).split():
            if ascii_sign == "M511x510S27034490x490":
                ascii_sign = "M511x510S2c734490x490"
            if ascii_sign == "M510x518S21005490x483":
                ascii_sign = "M510x518S2c105490x483"
            _sign = signwriting_to_image(ascii_sign, trust_box=False)
            if self.config.preprocess.scale_image:
                _sign = resize_and_center_image(_sign, target_width=self.config.preprocess.width, target_height=self.config.preprocess.height)
            else:
                _sign = center_image_on_white_background(_sign, target_width=self.config.preprocess.width, target_height=self.config.preprocess.height)
            if self.config.preprocess.invert_image:
                _sign = ImageOps.invert(_sign)
            if self.preprocess is not None:
                if isinstance(self.preprocess, Compose):
                    _sign = self.preprocess(_sign) ## Mirar com afegir el preprocess desde hyperparameter
                elif isinstance(self.preprocess, ConvNextImageProcessor):
                    _sign = self.preprocess(_sign, do_resize=False, return_tensors="pt")['pixel_values'].squeeze()         
            try:
                assert isinstance(_sign, torch.Tensor), "sign must be a torch tensor"
            except AssertionError as error:
                logger.error(error)
            try:
                assert list(_sign.shape) == [self.config.preprocess.channels, self.config.preprocess.width, self.config.preprocess.height], f"Expected a tensor of shape of ({self.config.preprocess.channels}, {self.config.preprocess.width}, {self.config.preprocess.height}), but a tensor of shape {_sign.shape} was given)"
            except AssertionError as error:
                logger.error(error)
            sign_arrays.append(_sign)
        signs_concatenated = torch.stack(sign_arrays, dim=0)
        return signs_concatenated

