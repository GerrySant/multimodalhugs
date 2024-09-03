from datasets import BuilderConfig
from pathlib import Path
from typing import Any, Union, Dict, Optional
from dataclasses import dataclass, field
from typing import Optional
from multimodal_embedder.data.utils import string_to_list

@dataclass
class PreprocessArguments:
    width: int = field(default=224, metadata={"help": "width of the images after preprocessing"})
    height: int = field(default=224, metadata={"help": "height of the images after preprocessing"})
    channels: int = field(default=3, metadata={"help": "number of channels of the images after preprocessing"})
    invert_frame: bool = field(default=True, metadata={"help": "If True, invert the image pixel values."})
    dataset_mean: Optional[str] = field(default="[0.9819646859188279, 0.9819646859188279, 0.9819646859188279]",
                                         metadata={"help": "Mean of the Pixels Values of the video frames in the train split. For normalization purposes."})
    dataset_std: Optional[str] = field(default="[0.12833405937294548, 0.12833405937294548, 0.12833405937294548]",
                                       metadata={"help": "Std of the Pixels Values of the video frames in the train split. For normalization purposes."})
    do_resize: bool = field(default=False, metadata={"help": "Whether to resize the image."})
    do_center_crop: bool = field(default=False, metadata={"help": "Whether to center crop the image."})
    do_rescale: bool = field(default=True, metadata={"help": "Whether to rescale the image."})
    do_normalize: bool = field(default=True, metadata={"help": "Whether to normalize the image."})


    def __init__(self, cfg=None, **kwargs):
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
    name: str = "MultimodalMTDataConfig"
    data_dir: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the signsbank_plus repository."})
    train_file_name: Optional[str] = field(default=None, metadata={"help": "Name of the train metadata file."})
    dev_file_name: Optional[str] = field(default=None, metadata={"help": "Name of the dev metadata file."})
    test_file_name: Optional[str] = field(default=None, metadata={"help": "Name of the test metadata file."})
    filter_empty_samples: bool = field(default=True, metadata={"help": "If True, it filters samples with an empty field."})
    shuffle: bool = field(default=True, metadata={"help": "If True, it shuffles samples in the dataset."})
    src_lang_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the tokenizer of the new source languages."})
    text_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the text tokenizer."})
    max_seq_length: int = field(default=512, metadata={"help": "Max length for tokenization purposes"})
    remove_unused_columns: bool = field(default=True, metadata={"help": "If True, it adapts the get_item() to work with remove_unused_columns=True."})

    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = getattr(cfg.data, 'data_dir', self.data_dir)
        self.train_file_name = getattr(cfg.data, 'train_file_name', self.train_file_name)
        self.dev_file_name = getattr(cfg.data, 'dev_file_name', self.dev_file_name)
        self.test_file_name = getattr(cfg.data, 'test_file_name', self.test_file_name)
        self.filter_empty_samples = getattr(cfg.data, 'filter_empty_samples', self.filter_empty_samples)
        self.shuffle = getattr(cfg.data, 'shuffle', self.shuffle)
        self.src_lang_tokenizer_path = getattr(cfg.data, 'src_lang_tokenizer_path', self.src_lang_tokenizer_path)
        self.text_tokenizer_path = getattr(cfg.data, 'text_tokenizer_path', self.text_tokenizer_path)
        self.max_seq_length = getattr(cfg.data, 'max_seq_length', self.max_seq_length)
        self.preprocess = PreprocessArguments(getattr(cfg.data, 'preprocess', None))
