import os
import math
import torch
import datasets
import pandas as pd

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from multimodalhugs.data import (
    MultimodalMTDataConfig,
    duration_filter,
)

class BilingualText2TextDataset(datasets.GeneratorBasedBuilder):
    def __init__(
        self,
        config: MultimodalMTDataConfig,
        info: Optional[DatasetInfo] = None,
        *args,
        **kwargs
    ):
        info = DatasetInfo(description="General Dataset class for bilingual translation datasets.") if info is None else info
        super().__init__(info=info, *args, **kwargs)
        self.config = config
    
    def _info(self):
        dataset_features = {
                "source": str,
                "source_prompt": Optional[str],
                "generation_prompt": Optional[str],
                "output_text": Optional[str],
            }
        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="General class for bilingual translation datasets",
            features=dataset_features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
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
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metafile_path": self.config.train_metadata_dir, 
                    "split": f"{datasets.Split.TRAIN}"
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

        for idx, item in enumerate(dataset):
            yield idx, {
                "source": item['source_signal'],
                "source_prompt": item['source_prompt'],
                "generation_prompt": item['generation_prompt'],
                "output_text": item['output_text'],
            }