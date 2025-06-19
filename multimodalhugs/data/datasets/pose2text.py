import os
import math
import torch
import datasets

from pathlib import Path
from pose_format import Pose
from pose_format.pose_body import EmptyPoseBody
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

from functools import lru_cache

@lru_cache(maxsize=1)
def read_pose(file_path: str) -> Pose:
    # This utility speeds up the checking of poses 
    # if the same pose is accessed multiple times sequentially
    with open(file_path, "rb") as pose_file:
        return Pose.read(pose_file, pose_body=EmptyPoseBody) 

@dataclass
class Pose2TextDataConfig(MultimodalDataConfig):
    """
    **Pose2TextDataConfig: Configuration class for the Pose-to-Text dataset.**

    This configuration class defines parameters for processing sign language 
    pose sequences in the Pose2Text dataset.
    """
    name: str = "Pose2TextDataConfig"
    reduce_holistic_poses: bool = field(
        default=True, 
        metadata={"help": "If True, it reduces holistic poses. See https://github.com/sign-language-processing/pose for more information."}
    )
    max_frames: Optional[int] = field(
        default=None, 
        metadata={"help": "Pose related samples larger than this value will be filtered"}
    )
    skip_frames_stride: Optional[int] = field(
        default=None,
        metadata={"help": "If specified, skips temporal tokens from each signal using the specified stride."}
    )
    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the Pose2TextDataConfig.**

        This constructor assigns configuration parameters based on the provided 
        `cfg` object, if available. If no configuration is given, it falls back 
        to default values.
        """
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super, **extra_args)

        # Assign new arguments from config if available
        self.reduce_holistic_poses = valid_config.get("reduce_holistic_poses", self.reduce_holistic_poses)
        self.max_frames = valid_config.get("max_frames", self.max_frames)
        self.skip_frames_stride = valid_config.get("skip_frames_stride", self.skip_frames_stride)

        # Store any remaining kwargs (not expected by dataclass)
        self._extra_args = extra_args


@register_dataset("pose2text")
class Pose2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **Pose2TextDataset: A dataset class for Pose-to-Text tasks.**

    This dataset class is designed for processing sign language pose sequences 
    and generating text representations. It leverages metadata files to structure 
    the data into train, validation, and test splits.

    Go to [Pose2TextDataConfig documentation](/docs/data/dataconfigs/Pose2TextDataConfig.md) to find out what arguments to put in the config.

    """
    def __init__(
        self,
        config: Optional[Pose2TextDataConfig] = None,
        *args,
        **kwargs
    ):
        """
        **Initialize the Pose2TextDataset.**

        **Args:**
        - `config` (Pose2TextDataConfig): The dataset configuration.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        You can pass either:
        - a config object (`Pose2TextDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(Pose2TextDataConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Dataset class for Pose2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "pose2text"
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
                "signal": str,
                "signal_start": Optional[int],
                "signal_end": Optional[int],
                "encoder_prompt": Optional[str],
                "decoder_prompt": Optional[str],
                "output": Optional[str],
            }

        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="Pose2TextDataset sign language related task dataset",
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

    def _generate_examples(self, split: str, metafile_path: str, **unused_kwargs):
        """
        **Generate dataset examples as (key, example) tuples.**

        This method:
        - Loads metadata from a `.csv` metafile.
        - Filters out missing files.
        - Reads pose sequences from binary files.
        - Filters samples based on duration constraints.

        **Args:**
        - `split` (str): The dataset split (`train`, `validation`, or `test`).
        - `metafile_path` (str): Path to the metadata file.

        **Yields:**
        - `Tuple[int, dict]`: Index and dictionary containing processed sample data.
        """

        # We always pass split="train" to load_dataset to directly load the TSV as a Dataset object.
        # This does NOT affect our actual split (train/val/test), which is determined by the `split` argument
        # passed from _split_generators. load_dataset requires split="train" to avoid returning a dict of splits.
        dataset = load_dataset('csv', data_files=str(metafile_path), split='train', delimiter="\t", num_proc=get_num_proc()) 

        def mapping_function(sample):
            """
            **Process each sample by reading the pose buffer and calculating duration.**

            **Args:**
            - `sample` (dict): A dictionary containing sample metadata.

            **Returns:**
            - `dict`: The updated sample with the pose data duration.
            """
            pose = read_pose(sample['signal'])
            sample['DURATION'] = pose.body.duration_in_frames(
                start_time=sample['signal_start'] or None, 
                end_time=sample['signal_end'] or None
            )

            return sample

        # Filter out samples where the file path does not exist
        dataset = dataset.filter(lambda sample: file_exists_filter('signal', sample), num_proc=get_num_proc())

        # Filter out long samples, except for the test set
        if split != "test" and self.max_frames is not None:
            # Apply to create the DURATION column 
            dataset = dataset.map(mapping_function, num_proc=get_num_proc())
            dataset = dataset.filter(lambda sample: duration_filter(self.max_frames, sample), num_proc=get_num_proc())

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item['signal'],
                "signal_start": item['signal_start'],
                "signal_end": item['signal_end'],
                "encoder_prompt": item.get("encoder_prompt") or "",
                "decoder_prompt": item.get("decoder_prompt") or "",
                "output": item['output'],
            }
