from datasets import BuilderConfig
from pathlib import Path
from typing import Any, Union, Dict, Optional
from dataclasses import dataclass, field
from typing import Optional
from multimodalhugs.data.utils import string_to_list, gather_appropriate_data_cfg, build_merged_omegaconf_config

@dataclass
class PreprocessArguments:
    """
    **PreprocessArguments: Configuration for image and video frame preprocessing.**

    This class defines the preprocessing parameters applied to images or video frames 
    before passing them into a model. It includes options for resizing, cropping, 
    normalization, and rescaling.

    """
    width: int = field(default=224, metadata={"help": "Target width (in pixels) for images/frames after preprocessing."})
    height: int = field(default=224, metadata={"help": "Target height (in pixels) for images/frames after preprocessing."})
    channels: int = field(default=3, metadata={"help": "Number of color channels in the images/frames (e.g., 3 for RGB)."})
    invert_frame: bool = field(default=False, metadata={"help": "If True, inverts pixel values for preprocessing."})
    dataset_mean: Optional[str] = field(default="[0.9819, 0.9819, 0.9819]",
                                         metadata={"help": "Mean pixel values for dataset normalization, specified as a list."})
    dataset_std: Optional[str] = field(default="[0.1283, 0.1283, 0.1283]",
                                       metadata={"help": "Standard deviation values for dataset normalization, specified as a list."})
    do_resize: bool = field(default=False, metadata={"help": "If True, resizes images/frames to the target width and height."})
    do_center_crop: bool = field(default=False, metadata={"help": "If True, applies center cropping to images/frames."})
    do_rescale: bool = field(default=True, metadata={"help": "If True, rescales pixel values to a fixed range (e.g., 0-1)."})
    do_normalize: bool = field(default=True, metadata={"help": "If True, normalizes pixel values using dataset mean and std."})

    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the PreprocessArguments class.**

        This constructor assigns preprocessing parameters based on `cfg`, 
        allowing for default values to be overridden.

        **Args:**
        - `cfg` (Optional[dict]): Configuration dictionary for preprocessing settings.
        - `**kwargs`: Additional keyword arguments for overriding specific settings.

        """
        self.width = getattr(cfg, 'width', self.width)
        self.height = getattr(cfg, 'height', self.height)
        self.channels = getattr(cfg, 'channels', self.channels)
        self.invert_frame = getattr(cfg, 'invert_frame', self.invert_frame)
        self.dataset_mean = string_to_list(getattr(cfg, 'dataset_mean', self.dataset_mean))
        self.dataset_std = string_to_list(getattr(cfg, 'dataset_std', self.dataset_std))
        self.do_resize = getattr(cfg, 'do_resize', self.do_resize)
        self.do_center_crop = getattr(cfg, 'do_center_crop', self.do_center_crop)
        self.do_rescale = getattr(cfg, 'do_rescale', self.do_rescale)
        self.do_normalize = getattr(cfg, 'do_normalize', self.do_normalize)


@dataclass
class MultimodalMTDataConfig(BuilderConfig):
    """
    **MultimodalMTDataConfig: Configuration for multimodal machine translation datasets.**

    This class defines parameters for handling dataset metadata, preprocessing, 
    tokenization, and data shuffling.

    """

    name: str = "MultimodalMTDataConfig"
    train_metadata_file: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the training dataset metadata file."})
    validation_metadata_file: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the validation dataset metadata file."})
    test_metadata_file: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the test dataset metadata file."})
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory if the dataset actor instance has already been created"})
    shuffle: bool = field(default=True, metadata={"help": "If True, shuffles the dataset samples."})
    new_vocabulary: Optional[str] = field(default=None, metadata={"help": "Path to a file containing new tokens for the tokenizer."})
    text_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the pre-trained text tokenizer."})
    remove_unused_columns: bool = field(default=True, metadata={"help": "If True, removes unused columns from the dataset for efficiency."})
    preprocess: Optional[PreprocessArguments] = field(default=None, metadata={"help": "Configuration for dataset-level preprocessing (e.g., resizing, normalization).", "extra_info": "Check [PreprocessArguments documentation](docs/data/dataconfigs/others/PreprocessArguments.md) to see which arguments are accepted."})

    def __init__(self, cfg=None, **kwargs):
        """
        **Initialize the MultimodalMTDataConfig class.**

        This constructor assigns dataset configuration parameters and 
        applies values from `cfg` if provided.

        **Args:**
        - `cfg` (Optional[dict]): Dictionary containing dataset configuration settings.
        - `**kwargs`: Additional keyword arguments to override default values.
        """
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(**extra_args)
        
        self.train_metadata_file = getattr(data_cfg, 'train_metadata_file', self.train_metadata_file)
        self.validation_metadata_file = getattr(data_cfg, 'validation_metadata_file', self.validation_metadata_file)
        self.test_metadata_file = getattr(data_cfg, 'test_metadata_file', self.test_metadata_file)
        self.dataset_dir = getattr(data_cfg, 'dataset_dir', self.dataset_dir)
        self.shuffle = getattr(data_cfg, 'shuffle', self.shuffle)
        self.new_vocabulary = getattr(data_cfg, 'new_vocabulary', self.new_vocabulary)
        self.text_tokenizer_path = getattr(data_cfg, 'text_tokenizer_path', self.text_tokenizer_path)
        self.preprocess = PreprocessArguments(getattr(data_cfg, 'preprocess', None))
