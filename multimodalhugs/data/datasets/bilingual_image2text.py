import os
import math
import torch
import datasets
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Any, Union, Dict, Optional
from datasets import load_dataset, Dataset, DatasetInfo, SplitGenerator, Features
from dataclasses import dataclass, field

from multimodalhugs.data import (
    MultimodalMTDataConfig,
    BilingualText2TextDataset,
    get_images
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset

@dataclass
class BilingualImage2textMTDataConfig(MultimodalMTDataConfig):
    """
    **BilingualImage2textMTDataConfig: Configuration for Bilingual Image-to-Text Machine Translation datasets.**

    This configuration class extends `MultimodalMTDataConfig` to support datasets 
    where the source input is an **image representation of text**, rather than raw text. 
    It includes additional parameters for font selection and image generation mode.
    """
    name: str = "BilingualImage2textMTDataConfig"
    font_path: Optional[str] = field(default=None, metadata={"help": "Path to the '.ttf' file that determines the Path to the .tff file which determines the typography used in the image generation"})
    as_numpy: Optional[bool] = field(default=False, metadata={"help": "If True, it creates the images when creating the dataset. If False, the image are created in an online manner."})
    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the BilingualImage2textMTDataConfig.**

        This method initializes the dataset configuration, optionally loading values 
        from a provided config object (`cfg`). If attributes are present in the config file, 
        they are assigned to the class.
        """
        super().__init__(cfg=cfg, **kwargs)
        # Assign new arguments from config if available
        self.font_path = getattr(cfg.data, 'font_path', self.font_path)
        self.as_numpy = getattr(cfg.data, 'as_numpy', self.as_numpy)

@register_dataset("bilingual_image2text")
class BilingualImage2TextDataset(BilingualText2TextDataset):
    """
    **BilingualImage2TextDataset: Dataset for Bilingual Image-to-Text Translation.**

    This dataset class extends `BilingualText2TextDataset`, where the source input 
    is an image instead of raw text. It supports different configurations for handling images.

    Go to [BilingualImage2textMTDataConfig documentation](/docs/data/dataconfigs/BilingualImage2textMTDataConfig.md) to find out what arguments to put in the config.
    """
    def __init__(
        self,
        config: BilingualImage2textMTDataConfig,
        *args,
        **kwargs
    ):
        """
        **Initialize the BilingualImage2TextDataset.**

        This constructor initializes the dataset with the specified configuration.

        **Args:**
        - `config` (BilingualImage2textMTDataConfig): Dataset configuration.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.
        """
        info = DatasetInfo(description="General Dataset class for bilingual image2Text translation datasets.")
        self.as_numpy = config.as_numpy
        super().__init__(config=config, info=info, *args, **kwargs)
    
    def _info(self):
        """
        **Get dataset information and feature structure.**

        Defines the dataset structure and feature types.

        **Returns:**
        - `DatasetInfo`: Object containing dataset metadata, including:
            - `description`: General dataset information.
            - `features`: Dictionary defining dataset schema.
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
        if self.as_numpy:
            dataset_features["source"] = np.ndarray
        else:
            dataset_features["source"] = str
        dataset_features = datasets.Features(dataset_features)
        return DatasetInfo(
            description="General class for bilingual image2Text translation datasets",
            features=dataset_features,
            supervised_keys=None,
        )

    def _generate_examples(self, **kwargs):
        """
        **Generate dataset examples as (key, example) tuples.**

        This method:
        - Loads metadata from a `.csv` metafile.
        - Generates images dynamically if `as_numpy` is enabled.
        - Yields processed dataset examples.

        **Args:**
        - `**kwargs`: Dictionary containing:
            - `metafile_path` (str): Path to the metadata file.
            - `split` (str): Dataset split (`train`, `validation`, or `test`).

        **Yields:**
        - `Tuple[int, dict]`: Index and dictionary containing processed sample data.
        """
        def create_image_secuences(sample):
            """
            **Generate an image representation of the text source.**

            This function converts the text source into an image using 
            the specified font and preprocessing parameters.

            **Args:**
            - `sample` (dict): Dictionary containing the source text.

            **Returns:**
            - `dict`: Updated sample with `source` replaced by an image representation.
            """
            sample['source'] = get_images(
                src_text=sample['source'],
                font_path=self.config.font_path,
                width=self.config.preprocess.width,
                height=self.config.preprocess.height,
                normalize_image=self.config.preprocess.do_normalize,
                mean=self.config.preprocess.dataset_mean,
                std=self.config.preprocess.dataset_std,
            )
            return sample

        metafile_path = kwargs['metafile_path']
        split = kwargs['split']

        dataset = load_dataset('csv', data_files=[str(metafile_path)], split="train", delimiter="\t", num_proc=get_num_proc())

        if self.as_numpy:
            dataset = dataset.map(create_image_secuences, num_proc=get_num_proc())

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "source": item.get('source', item['source_signal']),
                "source_start": item.get('start_time', 0),
                "source_end": item.get('end_time', 0),
                "source_prompt": item.get('source_prompt', ""),
                "generation_prompt": item.get('generation_prompt', ""),
                "output_text": item['output_text'],
            }