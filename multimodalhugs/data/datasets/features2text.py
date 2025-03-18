import os
import math
import torch
import datasets
import numpy as np


from pathlib import Path
from pose_format import Pose
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features
from dataclasses import dataclass, field

from multimodalhugs.data import (
    MultimodalMTDataConfig,
    contains_empty,
    file_exists_filter,
    duration_filter,
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset

@dataclass
class Features2TextDataConfig(MultimodalMTDataConfig):
    """
    **Features2TextDataConfig: Configuration class for the Feature-to-Text dataset.**

    This configuration class defines parameters for processing sign language 
    pose sequences in the Feature2Text dataset.
    """
    name: str = "Features2TextDataConfig"
    max_frames: Optional[int] = field(
        default=None, 
        metadata={"help": "Feature related samples larger than this value will be filtered"}
    )
    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the Features2TextDataConfig.**

        This constructor assigns configuration parameters based on the provided 
        `cfg` object, if available. If no configuration is given, it falls back 
        to default values.
        """
        super().__init__(cfg=cfg, **kwargs)
        # Assign new arguments from config if available
        self.max_frames = getattr(cfg.data, 'max_frames', self.max_frames)

@register_dataset("features2text")
class Features2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **Features2TextDataset: A dataset class for Feature-to-Text tasks.**

    This dataset class is designed for processing sign language features sequences 
    and generating text representations. It leverages metadata files to structure 
    the data into train, validation, and test splits.

    Go to [Features2TextDataConfig documentation](/docs/data/dataconfigs/Features2TextDataConfig.md) to find out what arguments to put in the config.

    """
    def __init__(
        self,
        config: Features2TextDataConfig, 
        *args,
        **kwargs
    ):
        """
        **Initialize the Features2TextDataset.**

        **Args:**
        - `config` (Features2TextDataConfig): The dataset configuration.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        """
        dataset_info = DatasetInfo(description="Dataset class for Features2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "feature2text"
        self.config = config
        self.max_frames = config.max_frames

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
                "source": str,
                "source_start": Optional[int],
                "source_end": Optional[int],
                "source_prompt": Optional[str],
                "generation_prompt": Optional[str],
                "output_text": Optional[str],
            }

        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="Features2TextDataset related task dataset",
            features=dataset_features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """
        **Define dataset splits based on metadata files.**

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
        - Filters out missing files.
        - Reads pose sequences from binary files.
        - Filters samples based on duration constraints.

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

        def mapping_function(sample):
            """
            **Process each sample by reading the features and extracting the duration.**

            **Args:**
            - `sample` (dict): A dictionary containing sample metadata.

            **Returns:**
            - `dict`: The updated sample with the features data duration.
            """

            sample['source'] = sample['source_signal']
            with open(sample['source'], "rb") as f:
                features = np.load(f)
                sample['DURATION'] = int(features.shape[0])
            return sample

        # Filter out samples where the file path does not exist
        dataset = dataset.filter(lambda sample: file_exists_filter('source_signal', sample), num_proc=get_num_proc())

        # Apply the update to the VIDEO_NAME column
        dataset = dataset.map(mapping_function, num_proc=get_num_proc())

        dataset = dataset.filter(lambda sample: duration_filter(self.max_frames, sample), num_proc=get_num_proc())

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "source": item['source'],
                "source_start": item.get("source_start") or 0,
                "source_end": item.get("source_end") or 0,
                "source_prompt": item.get("source_prompt") or "",
                "generation_prompt": item.get("generation_prompt") or "",
                "output_text": item['output_text'],
            }