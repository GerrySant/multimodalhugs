import os
import torch
import datasets

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from signwriting.tokenizer import normalize_signwriting
from multimodalhugs.data import (
    MultimodalDataConfig,
    check_columns,
    contains_empty,
    resolve_and_update_config,
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset
from multimodalhugs.custom_datasets import properly_format_signbank_plus

@register_dataset("signwriting")
class SignWritingDataset(datasets.GeneratorBasedBuilder):
    """
    **SignWritingDataset: A dataset class for SignWriting-based multimodal translation.**

    This dataset class processes SignWriting samples for multimodal machine translation tasks. 
    It loads structured datasets from metadata files and prepares examples for training, 
    validation, and testing.

    Go to [MultimodalDataConfig documentation](/docs/data/dataconfigs/MultimodalDataConfig.md) to find out what arguments to put in the config.
    """

    def __init__(
        self,
        config: Optional[MultimodalDataConfig] = None,
        *args,
        **kwargs
    ):
        """
        **Initialize the SignWritingDataset.**

        **Args:**
        - `config` (MultimodalDataConfig): Configuration object containing dataset parameters.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        You can pass either:
        - a config object (`MultimodalDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(MultimodalDataConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Custom dataset for SignWriting")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.config = config
        
    def _info(self):
        """
        **Get dataset information and feature structure.**

        **Returns:**
        - `DatasetInfo`: A dataset metadata object containing:
            - `description`: General dataset information.
            - `features`: The dataset schema with data types.
            - `supervised_keys`: `None` (no explicit supervised key pair).
        """
        dataset_features = {
                "signal": str,
                "signal_start": Optional[int],
                "signal_end": Optional[int],
                "encoder_prompt": Optional[str],
                "decoder_prompt": Optional[str],
                "output": Optional[str],
            }
        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="SignWriting Multimodal Machine Translation Dataset",
            features=dataset_features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """
        **Define dataset splits based on metadata files.**

        Reads metadata files and creates dataset splits for training, validation, and testing.

        **Args:**
        - `dl_manager` (DownloadManager): The dataset download manager (not used here).

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
        - Filters out samples that contain empty values.
        - Extracts relevant fields from the dataset.

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
        dataset = dataset.filter(lambda sample: not contains_empty(sample), num_proc=get_num_proc())

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item.get('signal', ''),
                "signal_start": item.get("start_time") or 0,
                "signal_end": item.get("end_time") or 0,
                "encoder_prompt": item.get("encoder_prompt") or "",
                "decoder_prompt": item.get("decoder_prompt") or "",
                "output": item['output'],
            }