from dataclasses import dataclass, field
from typing import Optional
from multimodal_embedder.data.utils import string_to_list

@dataclass
class PreprocessArguments:
    scale_image: bool = field(default=False, metadata={"help": "If True, Scale and centers the image."})
    width: int = field(default=224, metadata={"help": "width of the images after preprocessing"})
    height: int = field(default=224, metadata={"help": "height of the images after preprocessing"})
    channels: int = field(default=3, metadata={"help": "number of channels of the images after preprocessing"})
    invert_image: bool = field(default=True, metadata={"help": "If True, invert the image pixel values."})
    dataset_mean: Optional[str] = field(default="[0.9819646859188279, 0.9819646859188279, 0.9819646859188279]",
                                         metadata={"help": "Mean of the Pixels Values of the video frames in the train split. For normalization purposes."})
    dataset_std: Optional[str] = field(default="[0.12833405937294548, 0.12833405937294548, 0.12833405937294548]",
                                       metadata={"help": "Std of the Pixels Values of the video frames in the train split. For normalization purposes."})

    def __init__(self, cfg=None, **kwargs):
        self.scale_image = getattr(cfg, 'scale_image', self.scale_image)
        self.width = getattr(cfg, 'width', self.width)
        self.height = getattr(cfg, 'height', self.height)
        self.channels = getattr(cfg, 'channels', self.channels)
        self.invert_image = getattr(cfg, 'invert_image', self.invert_image)
        self.dataset_mean = string_to_list(getattr(cfg, 'dataset_mean', self.dataset_mean))
        self.dataset_std = string_to_list(getattr(cfg, 'dataset_std', self.dataset_std))

@dataclass
class MultimodalMTDataConfig:
    train_path: Optional[str] = field(default=None, metadata={"help": "Path to the train metadata file."})
    val_path: Optional[str] = field(default=None, metadata={"help": "Path to the val metadata file."})
    test_path: Optional[str] = field(default=None, metadata={"help": "Path to the test metadata file."})
    filter_empty_samples: bool = field(default=True, metadata={"help": "If True, it filters samples with an empty field."})
    shuffle: bool = field(default=True, metadata={"help": "If True, it shuffles samples in the dataset."})
    src_lang_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the tokenizer of the new source languages."})
    text_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the text tokenizer."})
    max_seq_length: int = field(default=512, metadata={"help": "Max length for tokenization purposes"})
    remove_unused_columns: bool = field(default=True, metadata={"help": "If True, it adapts the get_item() to work with remove_unused_columns=True."})

    def __init__(self, cfg=None, **kwargs):
        self.train_path = getattr(cfg.data, 'train_path', self.train_path)
        self.val_path = getattr(cfg.data, 'val_path', self.val_path)
        self.test_path = getattr(cfg.data, 'test_path', self.test_path)
        self.filter_empty_samples = getattr(cfg.data, 'filter_empty_samples', self.filter_empty_samples)
        self.shuffle = getattr(cfg.data, 'shuffle', self.shuffle)
        self.src_lang_tokenizer_path = getattr(cfg.data, 'src_lang_tokenizer_path', self.src_lang_tokenizer_path)
        self.text_tokenizer_path = getattr(cfg.data, 'text_tokenizer_path', self.text_tokenizer_path)
        self.max_seq_length = getattr(cfg.data, 'max_seq_length', self.max_seq_length)
        self.preprocess = PreprocessArguments(getattr(cfg.data, 'preprocess', None))
