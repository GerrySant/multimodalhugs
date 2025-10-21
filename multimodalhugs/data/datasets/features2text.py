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
    MultimodalDataConfig,
    contains_empty,
    file_exists_filter,
    duration_filter,
    resolve_and_update_config,
    gather_appropriate_data_cfg,
    get_all_dataclass_fields, 
    build_merged_omegaconf_config
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset

@dataclass
class Features2TextDataConfig(MultimodalDataConfig):
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
    preload_features: bool = field(
        default=False, 
        metadata={"help": "If True, the feature files are read at the dataset level instead of at the processor level."}
    )

    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the Features2TextDataConfig.**

        This constructor assigns configuration parameters based on the provided 
        `cfg` object, if available. If no configuration is given, it falls back 
        to default values.
        """
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super)

        # Set current class fields (in case parent didn’t)
        self.max_frames = valid_config.get("max_frames", self.max_frames)
        self.preload_features = valid_config.get("preload_features", self.preload_features)

        # Store any remaining kwargs (not expected by dataclass)
        self._extra_args = extra_args


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
        config: Optional[Features2TextDataConfig] = None,
        *args,
        **kwargs
    ):
        """
        Initialize the Features2TextDataset.

        You can pass either:
        - a config object (`Features2TextDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(Features2TextDataConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Dataset class for Features2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)
        self.name = "feature2text"
        self.config = config
        self.max_frames = config.max_frames
        self.preload_features = config.preload_features

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
                "signal": Union[str, np.ndarray],
                "signal_start": Optional[int],
                "signal_end": Optional[int],
                "encoder_prompt": Optional[str],
                "decoder_prompt": Optional[str],
                "output": Optional[str],
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
            features = np.load(sample['signal'])
            sample['DURATION'] = int(features.shape[0])

            if self.preload_features:
                sample['signal'] = np.array(features, dtype=np.float32)  # Ensure it remains a NumPy array
            else:
                sample['signal'] = sample['signal']
            return sample

        # Filter out samples where the file path does not exist
        dataset = dataset.filter(lambda sample: file_exists_filter('signal', sample), num_proc=get_num_proc())

        # Apply the update to the VIDEO_NAME column
        dataset = dataset.map(mapping_function, num_proc=get_num_proc())

        if self.max_frames is not None:
            dataset = dataset.filter(lambda sample: duration_filter(self.max_frames, sample), num_proc=get_num_proc())

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item['signal'],
                "signal_start": item.get("signal_start") or 0,
                "signal_end": item.get("signal_end") or 0,
                "encoder_prompt": item.get("encoder_prompt") or "",
                "decoder_prompt": item.get("decoder_prompt") or "",
                "output": item['output'],
            }