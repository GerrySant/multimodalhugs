from datasets import BuilderConfig
from pathlib import Path
from typing import Any, Union, Dict, Optional
from dataclasses import dataclass, field
from typing import Optional
from multimodalhugs.data.utils import string_to_list

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
    train_metadata_dir: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the train metadata file."})
    validation_metadata_dir: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the validation metadata file."})
    test_metadata_dir: Union[str, Path, Dict] = field(default=None, metadata={"help": "Path to the test metadata file."})
    filter_empty_samples: bool = field(default=True, metadata={"help": "If True, it filters samples with an empty field."})
    shuffle: bool = field(default=True, metadata={"help": "If True, it shuffles samples in the dataset."})
    src_lang_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the tokenizer of the new source languages."})
    new_task_tokens_dictionary_path: Optional[str] = field(default=None, metadata={"help": "Path to the dictionary file of the new tasks tokens to be included as tokens."})
    task: Optional[str] = field(default=None, metadata={"help": "Task token to be added to the samples of this dataset."})
    text_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to the text tokenizer."})
    max_seq_length: int = field(default=512, metadata={"help": "Max length for tokenization purposes"})
    remove_unused_columns: bool = field(default=True, metadata={"help": "If True, it adapts the get_item() to work with remove_unused_columns=True."})
    fps: Optional[int] = field(default=None, metadata={"help": "In case the dataset contains videos, specify the number of frames per second of the videos"})
    max_frames: Optional[int] = field(default=None, metadata={"help": "Video related samples larger than this value will be filtered"})

    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.train_metadata_dir = getattr(cfg.data, 'train_metadata_dir', self.train_metadata_dir)
        self.validation_metadata_dir = getattr(cfg.data, 'validation_metadata_dir', self.validation_metadata_dir)
        self.test_metadata_dir = getattr(cfg.data, 'test_metadata_dir', self.test_metadata_dir)
        self.filter_empty_samples = getattr(cfg.data, 'filter_empty_samples', self.filter_empty_samples)
        self.shuffle = getattr(cfg.data, 'shuffle', self.shuffle)
        self.src_lang_tokenizer_path = getattr(cfg.data, 'src_lang_tokenizer_path', self.src_lang_tokenizer_path)
        self.new_task_tokens_dictionary_path = getattr(cfg.data, 'new_task_tokens_dictionary_path', self.new_task_tokens_dictionary_path)
        self.task = getattr(cfg.data, 'task', self.task)
        self.text_tokenizer_path = getattr(cfg.data, 'text_tokenizer_path', self.text_tokenizer_path)
        self.max_seq_length = getattr(cfg.data, 'max_seq_length', self.max_seq_length)
        self.fps = getattr(cfg.data, 'fps', self.fps)
        self.max_frames = getattr(cfg.data, 'max_frames', self.max_frames)
        self.preprocess = PreprocessArguments(getattr(cfg.data, 'preprocess', None))


@dataclass
class SignLanguageMTDataConfig(MultimodalMTDataConfig):
    is_numpy_video: Optional[bool] = field(default=False, metadata={"help": "sample['source'] will point to preprocessed .npy video files."})
    is_pose: Optional[bool] = field(default=True, metadata={"help": "sample['source'] will point to .pose files."})
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        # Assign new arguments from config if available
        print(f"cfg.data: {cfg.data}")
        self.is_numpy_video = getattr(cfg.data, 'is_numpy_video', self.is_numpy_video)
        self.is_pose = getattr(cfg.data, 'is_pose', self.is_pose)
        assert self.is_numpy_video != self.is_pose, "is_numpy_video and is_pose cannot have the same value"
        assert self.is_numpy_video or self.is_pose, "At least one of is_numpy_video or is_pose must be True"

@dataclass
class BilingualImage2textMTDataConfig(MultimodalMTDataConfig):
    font_path: Optional[str] = field(default=None, metadata={"help": "Path to the '.ttf' file that determines the Path to the .tff file which determines the typography used in the image generation"})
    as_numpy: Optional[bool] = field(default=False, metadata={"help": "If True, it creates the images when creating the dataset. If False, the image are created in an online manner."})
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        # Assign new arguments from config if available
        self.font_path = getattr(cfg.data, 'font_path', self.font_path)
        self.as_numpy = getattr(cfg.data, 'as_numpy', self.as_numpy)