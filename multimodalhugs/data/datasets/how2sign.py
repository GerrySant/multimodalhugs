import os
import math
import torch
import datasets

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from multimodalhugs.data import (
    SignLanguageMTDataConfig,
    contains_empty,
    file_exists_filter,
    duration_filter,
)

class How2SignDataset(datasets.GeneratorBasedBuilder):
    def __init__(
        self,
        config: SignLanguageMTDataConfig, 
        *args,
        **kwargs
    ):
        dataset_info = DatasetInfo(description="Dataset class for How2Sign.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "how2sign_poses" if config.is_pose else "how2sign"
        self.config = config
        self.fps = config.fps if config.fps is not None else 24
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
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metafile_path": self.config.train_metadata_dir, 
                    "split": f"{datasets.Split.TRAIN}"
                }
            ),
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
        ]

    def _generate_examples(self, **kwargs):
        """
        Yields examples as (key, example) tuples.
        """
        metafile_path = kwargs['metafile_path']
        split = kwargs['split']
        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t")

        # Update VIDEO_NAME column with the full file path
        def update_file_name(sample):
            sample['DURATION'] = math.ceil((sample['source_end'] - sample['source_start']) * self.fps)
            if 'input_clip' in sample and sample['input_clip']:
                if not os.path.exists(sample['input_clip']):
                    if self.config.is_numpy_video:
                        sample['source'] = f"{metafile_path.split("text")[0].rstrip("/")}/rgb_front/numpy_videos/{sample['input_clip']}.npy"
                    elif self.config.is_pose:
                        sample['source'] = f"{metafile_path.split("text")[0].rstrip("/")}/rgb_front/pose_estimation/{sample['input_clip']}.pose"
                    else:
                        raise ValueError("At least one of is_numpy_video or is_pose must be True")
                else:
                    sample['source'] = sample['input_clip']
                sample['source_start'] = 0
                sample['source_end'] = 0
            else:
                if not os.path.exists(sample['input_pose']):
                    if self.config.is_numpy_video:
                        sample['source'] = f"{metafile_path.split('text')[0].rstrip('/')}/rgb_front/numpy_videos/{sample['input_pose']}.npy"
                    elif self.config.is_pose:
                        sample['source'] = f"{metafile_path.split('text')[0].rstrip('/')}/rgb_front/pose_estimation/{sample['input_pose']}.pose"
                    else:
                        raise ValueError("At least one of is_numpy_video or is_pose must be True")
                else:
                    sample['source'] = sample['input_pose']
            return sample

        # Apply the update to the VIDEO_NAME column
        dataset = dataset.map(update_file_name)

        # Filter out samples where the updated file path does not exist
        dataset = dataset.filter(lambda sample: file_exists_filter('source', sample))

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