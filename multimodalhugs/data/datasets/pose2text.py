import os
import math
import torch
import datasets

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
class Pose2TextDataConfig(MultimodalMTDataConfig):
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
    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the Pose2TextDataConfig.**

        This constructor assigns configuration parameters based on the provided 
        `cfg` object, if available. If no configuration is given, it falls back 
        to default values.
        """
        super().__init__(cfg=cfg, **kwargs)
        # Assign new arguments from config if available
        self.reduce_holistic_poses = getattr(cfg.data, 'reduce_holistic_poses', self.reduce_holistic_poses)
        self.max_frames = getattr(cfg.data, 'max_frames', self.max_frames)

@register_dataset("pose2text")
class Pose2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **Pose2TextDataset: A dataset class for Pose-to-Text tasks.**

    This dataset class is designed for processing sign language pose sequences 
    and generating text representations. It leverages metadata files to structure 
    the data into train, validation, and test splits.

    Go to [Pose2TextDataConfig documentation](multimodalhugs/docs/data/dataconfigs/Pose2TextDataConfig.md) to find out what arguments to put in the config.

    """
    def __init__(
        self,
        config: Pose2TextDataConfig, 
        *args,
        **kwargs
    ):
        """
        **Initialize the Pose2TextDataset.**

        **Args:**
        - `config` (Pose2TextDataConfig): The dataset configuration.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        """
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
                "source": str,
                "source_start": Optional[int],
                "source_end": Optional[int],
                "source_prompt": Optional[str],
                "generation_prompt": Optional[str],
                "output_text": Optional[str],
            }

        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="How2Sign sign language related task dataset",
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

    _last_buffer = None
    _last_file_path = None
    
    def _read_pose(self, file_path):
        """
        **Read and cache the pose buffer from a file.**

        If the same file is requested sequentially, reuse the cached buffer to optimize performance.

        **Args:**
        - `file_path` (str): Path to the pose data file.

        **Returns:**
        - `bytes`: The binary pose data buffer.
        """
        if self._last_file_path != file_path:
            with open(file_path, "rb") as pose_file:
                buffer = pose_file.read()
                self._last_buffer = buffer
                self._last_file_path = file_path
        return self._last_buffer

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
            **Process each sample by reading the pose buffer and calculating duration.**

            **Args:**
            - `sample` (dict): A dictionary containing sample metadata.

            **Returns:**
            - `dict`: The updated sample with the pose data duration.
            """
            sample['source'] = sample['source_signal']
            
            buffer = self._read_pose(sample['source'])
            if (sample['source_end'] - sample['source_start']) == 0:
                pose = Pose.read(buffer) # [t, people, d, xyz]
            else:
                pose = Pose.read(buffer, start_time=sample['source_start'], end_time=sample['source_end']) # [t, people, d, xyz]
            sample['DURATION'] = len(pose.body.data)

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
                "source_start": item['source_start'],
                "source_end": item['source_end'],
                "source_prompt": item['source_prompt'] if item.get('source_prompt', "") is not None else "",
                "generation_prompt": item.get('generation_prompt', "") if item.get('generation_prompt', "") is not None else "",
                "output_text": item['output_text'],
            }
