import os
import math
import torch
import datasets
import pandas as pd

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from multimodalhugs.data import (
    BilingualMTDataConfig,
    duration_filter,
)

class BilingualText2TextDataset(datasets.GeneratorBasedBuilder):
    def __init__(
        self,
        config: BilingualMTDataConfig,
        info: Optional[DatasetInfo] = None,
        *args,
        **kwargs
    ):
        info = DatasetInfo(description="General Dataset class for bilingual translation datasets.") if info is None else info
        super().__init__(info=info, *args, **kwargs)
        self.config = config

    def _get_dataframe(self, split_directory, src_lang, tgt_lang):
        src_path = f"{split_directory}/{src_lang}.txt"
        tgt_path = f"{split_directory}/{tgt_lang}.txt"
    
        # Load both Hebrew and English text files
        src_text = open(src_path, "r", encoding="utf-8").readlines()
        tgt_text = open(tgt_path, "r", encoding="utf-8").readlines()
        
        # Create a DataFrame for this split
        df = pd.DataFrame({"source": src_text, "target": tgt_text})
        return Dataset.from_pandas(df)
    
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
            description="General class for bilingual translation datasets",
            features=dataset_features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_directory": os.path.join(self.config.data_dir, self.config.train_split_name), 
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split_directory": os.path.join(self.config.data_dir,  self.config.dev_split_name), 
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split_directory": os.path.join(self.config.data_dir,  self.config.test_split_name), 
                }
            ),
        ]

    def _generate_examples(self, **kwargs):
        """
        Yields examples as (key, example) tuples.
        """
        dataset = self._get_dataframe(
            split_directory=kwargs['split_directory'],
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang
            )

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "src_lang": self.config.src_lang,
                "source": item['source'],
                "tgt_lang": self.config.tgt_lang,
                "tgt_sentence": item['target'],
                "task": self.config.task,
            }