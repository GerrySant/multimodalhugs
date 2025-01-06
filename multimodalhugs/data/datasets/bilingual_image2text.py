import os
import math
import torch
import datasets
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from multimodalhugs.data import (
    BilingualImage2textMTDataConfig,
    BilingualText2TextDataset,
    get_images
)

class BilingualImage2TextDataset(BilingualText2TextDataset):
    def __init__(
        self,
        config: BilingualImage2textMTDataConfig,
        *args,
        **kwargs
    ):
        info = DatasetInfo(description="General Dataset class for bilingual image2Text translation datasets.")
        self.as_numpy = config.as_numpy
        super().__init__(config=config, info=info, *args, **kwargs)
    
    def _info(self):
        dataset_features = {
                "source": str,
                "source_start": Optional[float],
                "source_end": Optional[float],
                "source_prompt": Optional[str],
                "generation_prompt": Optional[str],
                "output_text": Optional[str],
            }
        if self.as_numpy:
            dataset_features["source"] = np.ndarray
        else:
            dataset_features["source"] = str
        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="General class for bilingual image2Text translation datasets",
            features=dataset_features,
            supervised_keys=None,
        )

    def _generate_examples(self, **kwargs):
        """
        Yields examples as (key, example) tuples.
        """
        def create_image_secuences(sample):
            sample['source'] = get_images(
                src_text=sample['source'],
                font_path=self.config.font_path,
                width=self.config.preprocess.width,
                height=self.config.preprocess.height,
                normalize_image=self.config.preprocess.do_normalize,
                mean=self.config.preprocess.dataset_mean,
                std=self.config.preprocess.dataset_std,
            )
            return sample

        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t")

        if self.as_numpy:
            dataset = dataset.map(create_image_secuences)

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "source": item.get('source', item['source_text']),
                "source_start": item.get('start_time', 0),
                "source_end": item.get('end_time', 0),
                "source_prompt": item.get('source_prompt', ""),
                "generation_prompt": item.get('generation_prompt', ""),
                "output_text": item['output_text'],
            }