import os
import torch
import datasets

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features

from signwriting.tokenizer import normalize_signwriting
from multimodalhugs.data import (
    SignLanguageMTDataConfig,
    contains_empty,
    file_exists_filter,
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
        
        self.data_dir = Path(config.data_dir) if type(config.data_dir) != Path else config.data_dir
        self.data_dir = self.data_dir / 'sentence_level' if self.data_dir.name != "sentence_level" else self.data_dir
        assert self.data_dir.is_dir(), f"Error: The {str(self.data_dir.name)} directory not found at {str(self.data_dir.parent)}."
        
    def _info(self):
        return DatasetInfo(
            description="How2Sign sign language related task dataset",
            features=datasets.Features({
                "src_lang": str,
                "source": str,
                "tgt_lang": str,
                "tgt_sentence": str,
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"train/text/en/raw_text/how2sign_train.csv"), 
                    "split": f"{datasets.Split.TRAIN}"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"val/text/en/raw_text/how2sign_val.csv"), 
                    "split": "val"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metafile_path": os.path.join(self.data_dir, f"test/text/en/raw_text/how2sign_test.csv"), 
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
            if self.config.is_numpy_video:
                sample['SENTENCE_NAME'] = f"{self.data_dir}/{split}/rgb_front/numpy_videos/{sample['SENTENCE_NAME']}.npy"
            elif self.config.is_pose:
                sample['SENTENCE_NAME'] = f"{self.data_dir}/{split}/rgb_front/pose_estimation/{sample['SENTENCE_NAME']}.pose"
            else:
                raise ValueError("At least one of is_numpy_video or is_pose must be True")
            return sample

        # Apply the update to the VIDEO_NAME column
        dataset = dataset.map(update_file_name)
        print(f"before_filter: {len(dataset)}")

        # Filter out samples where the updated file path does not exist
        dataset = dataset = dataset.filter(lambda sample: not contains_empty(sample))
        print(f"filter_1: {len(dataset)}")

        
        dataset = dataset.filter(lambda sample: file_exists_filter('SENTENCE_NAME', sample))
        print(f"filter_2: {len(dataset)}")

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "src_lang": 'asl',
                "source": item['SENTENCE_NAME'],
                "tgt_lang": 'en',
                "tgt_sentence": item['SENTENCE'],
            }