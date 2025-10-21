import os
import math
import torch
import datasets
import pandas as pd

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features
from dataclasses import dataclass, field

from multimodalhugs.data import (
    MultimodalDataConfig,
    duration_filter,
    gather_appropriate_data_cfg,
    resolve_and_update_config,
    build_merged_omegaconf_config
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset

@dataclass
class BilingualText2textMTDataConfig(MultimodalDataConfig):
    """
    **BilingualText2textMTDataConfig: Configuration for Bilingual Text-to-Text Machine Translation datasets.**

    This configuration class extends `MultimodalDataConfig` to support datasets 
    where the signal input is an **text representation of text**, rather than raw text. 
    It includes additional parameters for font selection and text generation mode.
    """
    name: str = "BilingualText2textMTDataConfig"
    max_source_tokens: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum number of tokens allowed to the source. Samples surpassing this limit will be droped"}
    )
    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the BilingualText2textMTDataConfig.**

        This method initializes the dataset configuration, optionally loading values 
        from a provided config object (`cfg`). If attributes are present in the config file, 
        they are assigned to the class.
        """
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super)

        # Set current class fields (in case parent didn’t)
        self.max_source_tokens = valid_config.get("max_source_tokens", self.max_source_tokens)

        # Store any remaining kwargs (not expected by dataclass)
        self._extra_args = extra_args

@register_dataset("bilingual_text2text")
class BilingualText2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **BilingualText2TextDataset: A dataset class for bilingual text-to-text translation.**

    This dataset class is designed for handling bilingual translation datasets 
    where text input in one language is mapped to its corresponding translation.

    Go to [BilingualText2textMTDataConfig documentation](/docs/data/dataconfigs/BilingualText2textMTDataConfig.md) to find out what arguments to put in the config.

    """
    def __init__(
        self,
        config: Optional[BilingualText2textMTDataConfig] = None,
        info: Optional[DatasetInfo] = None,
        *args,
        **kwargs
    ):
        """
        **Initialize the BilingualText2TextDataset.**

        **Args:**
        - `config` (BilingualText2textMTDataConfig): The dataset configuration containing metadata file paths.
        - `info` (Optional[DatasetInfo], default=`None`): Dataset metadata. If `None`, 
          a default `DatasetInfo` object is created.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        You can pass either:
        - a config object (`BilingualText2textMTDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(BilingualText2textMTDataConfig, config, kwargs)
        info = DatasetInfo(description="General Dataset class for bilingual translation datasets.") if info is None else info
        super().__init__(info=info, *args, **kwargs)
        self.config = config
        self.max_source_tokens = getattr(self.config, "max_source_tokens", None)
    
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

        def mapping_function(sample):
            sample['DURATION'] = len(str(sample['signal']).split())
            return sample

        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t", num_proc=get_num_proc())

        # Filter samples where the signal length is greater than self.max_source_tokens
        dataset = dataset.map(mapping_function, num_proc=get_num_proc())
        if self.max_source_tokens is not None:
            dataset = dataset.filter(lambda sample: duration_filter(self.max_source_tokens, sample), num_proc=get_num_proc())

        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item['signal'],
                "encoder_prompt": item.get("encoder_prompt") or "",
                "decoder_prompt": item.get("decoder_prompt") or "",
                "output": item['output'],
            }