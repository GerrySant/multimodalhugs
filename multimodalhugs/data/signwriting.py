import os
import torch
import datasets

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from signwriting.tokenizer import normalize_signwriting
from multimodalhugs.data import (
    SignLanguageMTDataConfig,
    contains_empty,
)

from signwriting.tokenizer import normalize_signwriting
from multimodalhugs.data import (
    MultimodalMTDataConfig,
    check_columns,
    contains_empty,
)
from multimodalhugs.custom_datasets import properly_format_signbank_plus

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
        self.train_split_name = config.train_split_name.split('.')[0] if config.train_split_name is not None else "corrected_train"
        self.dev_split_name = config.dev_split_name.split('.')[0] if config.dev_split_name is not None else "corrected_dev"
        self.test_split_name = config.test_split_name.split('.')[0] if config.test_split_name is not None else "corrected_all"
        
    def _info(self):
        dataset_features = {
                "src_lang": str,
                "source": str,
                "tgt_lang": str,
                "tgt_sentence": str,
                "task": Optional[str],
            }
        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="SignWriting Multimodal Machine Translation Dataset",
            features=dataset_features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"data/parallel/cleaned/{self.train_split_name}.csv"), 
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"data/parallel/cleaned/{self.dev_split_name}.csv"), 
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"data/parallel/test/{self.test_split_name}.csv"), 
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
                "task": self.config.task,
            }