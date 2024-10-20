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
        if self.as_numpy:
            dataset_features = datasets.Features({
                "src_lang": str,
                "source": np.ndarray,
                "tgt_lang": str,
                "tgt_sentence": str,
            })
        else:
            dataset_features = datasets.Features({
                "src_lang": str,
                "source": str,
                "tgt_lang": str,
                "tgt_sentence": str,
            })
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

        dataset = self._get_dataframe(
            split_directory=kwargs['split_directory'],
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang
            )

        if self.as_numpy:
            dataset = dataset.map(create_image_secuences)

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "src_lang": 'v' + self.config.src_lang,
                "source": item['source'],
                "tgt_lang": self.config.tgt_lang,
                "tgt_sentence": item['target'],
            }