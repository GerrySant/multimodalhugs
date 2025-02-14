import os
import math
import torch
import datasets

from pathlib import Path
from pose_format import Pose
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from multimodalhugs.data import (
    Pose2TextDataConfig,
    contains_empty,
    file_exists_filter,
    duration_filter,
)

class Pose2TextDataset(datasets.GeneratorBasedBuilder):
    def __init__(
        self,
        config: Pose2TextDataConfig, 
        *args,
        **kwargs
    ):
        dataset_info = DatasetInfo(description="Dataset class for Pose2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "pose2text"
        self.config = config
        self.max_frames = config.max_frames

    def _info(self):
        dataset_features = {
                "source": str,
                "source_start": Optional[float],
                "source_end": Optional[float],
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
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metafile_path": self.config.validation_metadata_dir, 
                    "split": "val"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metafile_path": self.config.test_metadata_dir, 
                    "split": f"{datasets.Split.TEST}"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metafile_path": self.config.train_metadata_dir, 
                    "split": f"{datasets.Split.TRAIN}"
                }
            ),
        ]

    _last_buffer = None
    __last_file_path = None
    
    def _read_pose(self, file_path):
        """
        Reads and caches the pose buffer from a file.
        If the same file is requested sequentially, reuse the cached buffer.
        """
        if self._last_file_path != file_path:
            with open(file_path, "rb") as pose_file:
                buffer = pose_file.read()
                self._last_buffer = buffer
                self._last_file_path = file_path
        return self._last_buffer

    def _generate_examples(self, **kwargs):
        """
        Yields examples as (key, example) tuples.
        """
        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t")

        def mapping_function(sample):
            sample['source'] = sample['source_signal']
            
            buffer = self._read_pose(sample['source'])
            if (sample['source_end'] - sample['source_start']) == 0:
                pose = Pose.read(buffer) # [t, people, d, xyz]
            else:
                pose = Pose.read(buffer, start_time=sample['source_start'], end_time=sample['source_end']) # [t, people, d, xyz]
            sample['DURATION'] = len(pose.body.data)

            return sample

        # Apply the update to the VIDEO_NAME column
        dataset = dataset.map(mapping_function)
        # Filter out samples where the updated file path does not exist
        dataset = dataset.filter(lambda sample: file_exists_filter('source', sample))

        # dataset = dataset.map(obtain_duration)

        dataset = dataset.filter(lambda sample: duration_filter(self.max_frames, sample))

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
