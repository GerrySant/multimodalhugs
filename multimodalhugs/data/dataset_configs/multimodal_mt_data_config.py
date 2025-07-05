from datasets import BuilderConfig
from pathlib import Path
from typing import Any, Union, Dict, Optional
from dataclasses import dataclass, field
from typing import Optional
from multimodalhugs.data.utils import gather_appropriate_data_cfg, build_merged_omegaconf_config

@dataclass
class MultimodalDataConfig(BuilderConfig):
    """
    **MultimodalDataConfig: Configuration for multimodal machine translation datasets.**

    This class defines parameters for handling dataset metadata, preprocessing, 
    tokenization, and data shuffling.

    """

    name: str = "MultimodalDataConfig"
    train_metadata_file: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the training dataset metadata file."})
    validation_metadata_file: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the validation dataset metadata file."})
    test_metadata_file: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the test dataset metadata file."})
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory if the dataset actor instance has already been created"})
    shuffle: bool = field(default=True, metadata={"help": "If True, shuffles the dataset samples."})
    remove_unused_columns: bool = field(default=True, metadata={"help": "If True, removes unused columns from the dataset for efficiency."})

    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the MultimodalDataConfig class.**

        This constructor assigns dataset configuration parameters and 
        applies values from `cfg` if provided.

        **Args:**
        - `cfg` (Optional[dict]): Dictionary containing dataset configuration settings.
        - `**kwargs`: Additional keyword arguments to override default values.
        """
        data_cfg = gather_appropriate_data_cfg(cfg)
        _, extra_args, _ = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(**extra_args)
        
        self.train_metadata_file = getattr(data_cfg, 'train_metadata_file', self.train_metadata_file)
        self.validation_metadata_file = getattr(data_cfg, 'validation_metadata_file', self.validation_metadata_file)
        self.test_metadata_file = getattr(data_cfg, 'test_metadata_file', self.test_metadata_file)
        self.dataset_dir = getattr(data_cfg, 'dataset_dir', self.dataset_dir)
        self.shuffle = getattr(data_cfg, 'shuffle', self.shuffle)