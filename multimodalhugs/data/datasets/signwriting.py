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
        
    def _info(self):
        dataset_features = {
                "source": str,
                "source_start": Optional[float],
                "source_end": Optional[float],
                "source_prompt": Optional[str],
                "generation_prompt": Optional[str],
                "output_text": Optional[str],
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
                    "metafile_path": self.config.train_metadata_dir, 
                    "split": f"{datasets.Split.TRAIN}"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metafile_path": self.config.validation_metadata_dir, 
                    "split": "val"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metafile_path": self.config.test_metadata_dir, 
                    "split": f"{datasets.Split.TEST}"
                }
            ),
        ]

    def _generate_examples(self, **kwargs):
        """
        Yields examples as (key, example) tuples.
        """
        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t")
        dataset = dataset.filter(lambda sample: not contains_empty(sample))

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "source": item.get('source_signal', ''),
                "source_start": item.get('start_time', 0),
                "source_end": item.get('end_time', 0),
                "source_prompt": item.get('source_prompt', ""),
                "generation_prompt": item.get('generation_prompt', ""),
                "output_text": item['output_text'],
            }