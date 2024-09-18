import os
from pathlib import Path
from typing import Any, Union, Dict, Optional
import logging

import torch
from PIL import ImageOps
from datasets import load_dataset, Dataset
from torchvision.transforms import Compose
from transformers import M2M100Tokenizer, PreTrainedTokenizerFast

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image
from multimodalhugs.data import (
    MultimodalMTDataConfig,
    check_columns,
    load_tokenizer_from_vocab_file,
    pad_and_create_mask,
    center_image_on_white_background,
    contains_empty,
)
from multimodalhugs.custom_datasets import properly_format_signbank_plus
from multimodalhugs.data.utils import _transform

import argparse
from pathlib import Path
from PIL import Image

import datasets
from datasets import DatasetBuilder, DatasetInfo

REQUIRED_COLUMNS = ['source', 'target', 'tgt_lang', 'src_lang']

def ensure_manifest_format(metafile_path):
    metafile_path = Path(metafile_path)
    if "corrected_" in metafile_path.name:
        return metafile_path
    else:
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train")
        if not check_columns(dataset, REQUIRED_COLUMNS):
            dir_name, file_name = os.path.split(metafile_path)
            new_file_name = "corrected_" + file_name
            out_path = Path(dir_name) / new_file_name
            if not out_path.exists():
                properly_format_signbank_plus(metafile_path)
            return out_path
        else:
            return metafile_path

class SignWritingDataset(datasets.GeneratorBasedBuilder):
    def __init__(
        self,
        config: MultimodalMTDataConfig, 
        *args,
        **kwargs
    ):
        dataset_info = DatasetInfo(description="Custom dataset for SignWriting")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.config = config
        self.data_dir = config.data_dir
        self.train_file_name = config.train_file_name.split('.')[0] if config.train_file_name is not None else "corrected_train"
        self.dev_file_name = config.dev_file_name.split('.')[0] if config.dev_file_name is not None else "corrected_dev"
        self.test_file_name = config.test_file_name.split('.')[0] if config.test_file_name is not None else "corrected_all"
        
    def _info(self):
        return DatasetInfo(
            description="SignWriting Multimodal Machine Translation Dataset",
            features=datasets.Features({
                "src_lang": str,
                "source": str,
                "tgt_lang": str,
                "tgt_sentence": str,
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"data/parallel/cleaned/{self.train_file_name}.csv"), 
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"data/parallel/cleaned/{self.dev_file_name}.csv"), 
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"data/parallel/test/{self.test_file_name}.csv"), 
                    "split": "test"
                }
            ),
        ]

    def _generate_examples(self, **kwargs):
        """
        Yields examples as (key, example) tuples.
        """
        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train")
        if not check_columns(dataset, REQUIRED_COLUMNS):
            dir_name, file_name = os.path.split(metafile_path)
            new_file_name = "corrected_" + file_name
            out_path = Path(dir_name) / new_file_name
            if not out_path.exists():
                properly_format_signbank_plus(metafile_path)
            dataset = load_dataset('csv', data_files=[str(out_path)], split="train")

        dataset = dataset.filter(lambda sample: not contains_empty(sample))

        for idx, item in enumerate(dataset):

            yield idx, {
                "src_lang": item['src_lang'],
                "source": item['source'],
                "tgt_lang": item['tgt_lang'],
                "tgt_sentence": item['target'],
            }