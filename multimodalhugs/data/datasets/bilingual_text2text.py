import os
import math
import torch
import datasets
import pandas as pd

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from multimodalhugs.data import (
    MultimodalDataConfig,
    duration_filter,
    resolve_and_update_config,
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset

@register_dataset("bilingual_text2text")
class BilingualText2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **BilingualText2TextDataset: A dataset class for bilingual text-to-text translation.**

    This dataset class is designed for handling bilingual translation datasets 
    where text input in one language is mapped to its corresponding translation.

    Go to [MultimodalDataConfig documentation](/docs/data/dataconfigs/MultimodalDataConfig.md) to find out what arguments to put in the config.

    """
    def __init__(
        self,
        config: Optional[MultimodalDataConfig] = None,
        info: Optional[DatasetInfo] = None,
        *args,
        **kwargs
    ):
        """
        **Initialize the BilingualText2TextDataset.**

        **Args:**
        - `config` (MultimodalDataConfig): The dataset configuration containing metadata file paths.
        - `info` (Optional[DatasetInfo], default=`None`): Dataset metadata. If `None`, 
          a default `DatasetInfo` object is created.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        You can pass either:
        - a config object (`MultimodalDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(MultimodalDataConfig, config, kwargs)
        info = DatasetInfo(description="General Dataset class for bilingual translation datasets.") if info is None else info
        super().__init__(info=info, *args, **kwargs)
        self.config = config
    
    def _info(self):
        """
        **Get dataset information and feature structure.**

        Defines the expected structure of the dataset, including input and output text fields.

        **Returns:**
        - `DatasetInfo`: A dataset metadata object containing:
            - `description`: General dataset information.
            - `features`: The dataset schema with data types.
            - `supervised_keys`: `None` (no explicit supervised key pair).
        """
        dataset_features = {
                "signal": str,
                "encoder_prompt": Optional[str],
                "decoder_prompt": Optional[str],
                "output": Optional[str],
            }
        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="General class for bilingual translation datasets",
            features=dataset_features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """
        **Define dataset splits based on metadata files.**

        **Args:**
        - `dl_manager` (DownloadManager): The dataset download manager (not used here since data is local).

        **Returns:**
        - `List[datasets.SplitGenerator]`: A list of dataset splits (`train`, `validation`, `test`).
        """
        splits = []
        if self.config.train_metadata_file is not None:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "metafile_path": self.config.train_metadata_file,
                        "split": "train"
                    }
                )
            )
        if self.config.validation_metadata_file is not None:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "metafile_path": self.config.validation_metadata_file,
                        "split": "validation"
                    }
                )
            )
        if self.config.test_metadata_file is not None:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "metafile_path": self.config.test_metadata_file,
                        "split": "test"
                    }
                )
            )
        return splits

    def _generate_examples(self, **kwargs):
        """
        **Generate dataset examples as (key, example) tuples.**

        This method:
        - Loads metadata from a `.csv` metafile.
        - Iterates through each sample and extracts relevant text fields.

        **Args:**
        - `**kwargs`: Dictionary containing:
            - `metafile_path` (str): Path to the metadata file.
            - `split` (str): The dataset split (`train`, `validation`, or `test`).

        **Yields:**
        - `Tuple[int, dict]`: Index and dictionary containing processed sample data.
        """
        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t", num_proc=get_num_proc())

        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item['signal'],
                "encoder_prompt": item.get("encoder_prompt") or "",
                "decoder_prompt": item.get("decoder_prompt") or "",
                "output": item['output'],
            }