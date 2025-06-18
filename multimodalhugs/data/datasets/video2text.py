import os
import av
import torch
import datasets
from pathlib import Path
from omegaconf import ListConfig
from typing import Any, Union, Optional, Dict, Tuple
from dataclasses import dataclass, field

from torchvision.io import read_video
from datasets import DatasetInfo, SplitGenerator
from datasets import load_dataset

from multimodalhugs.data import (
    MultimodalDataConfig,
    file_exists_filter,
    duration_filter,
    resolve_and_update_config,
    gather_appropriate_data_cfg,
    get_all_dataclass_fields, 
    build_merged_omegaconf_config
)

from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset

@dataclass
class Video2TextDataConfig(MultimodalDataConfig):
    """
    Configuration for Video-to-Text dataset.

    Args:
        name (str): Identifier for this config class.
        max_frames (Optional[int]): Filter out videos longer than this many frames.
        normalize (bool): If True, scales pixel values to [0…1].
        resize (Optional[int | Tuple[int, int]]):
            - int → square resize (H=W=resize)
            - tuple (H, W) → resize to that shape
            - list [H, W] in your YAML will be auto-converted to tuple
            - None → keep original size
    """
    name: str = "Video2TextDataConfig"
    max_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out videos longer than this (in frames)."}
    )
    custom_preprocessor_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to an Autoprocessor from HuggingFace"}
    )
    join_chw: bool = field(
        default=False,
        metadata={"help": "if True, it returns a tensor of torch.Size([B, T, C*H*W]). If False, it returns a tensor of torch.Size([B, T, C, H, W])."}
    )
    skip_frames_stride: Optional[int] = field(
        default=None,
        metadata={"help": "If specified, skips temporal tokens from each signal using the specified stride."}
    )
    def __init__(self, cfg=None, **kwargs):
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super, **extra_args)
        # pull from OmegaConf yaml (or leave defaults)
        self.max_frames = valid_config.get("max_frames", self.max_frames)
        self.custom_preprocessor_path = valid_config.get("custom_preprocessor_path", self.custom_preprocessor_path)
        self.join_chw = valid_config.get("join_chw", self.join_chw)
        self.skip_frames_stride = valid_config.get("skip_frames_stride", self.skip_frames_stride)


@register_dataset("video2text")
class Video2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **Video2TextDataset: A dataset class for Video-to-Text tasks.**
    """
    def __init__(
        self, 
        config: Optional[Video2TextDataConfig] = None,
        *args, 
        **kwargs
    ):
        """
        Initialize the Video2TextDataset.

        You can pass either:
        - a config object (`Video2TextDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(Video2TextDataConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Dataset class for Video2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "video2text"
        self.config = config
        self.max_frames = config.max_frames

    def _info(self):
        features = {
            "signal": str,
            "signal_start": Optional[int],
            "signal_end": Optional[int],
            "encoder_prompt": Optional[str],
            "decoder_prompt": Optional[str],
            "output": Optional[str],
        }
        return DatasetInfo(
            description="Video2TextDataset for multimodal sequence-to-text",
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager) -> list:
        splits = []
        if self.config.train_metadata_file:
            splits.append(
                SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "metafile_path": self.config.train_metadata_file,
                        "split": "train",
                    },
                )
            )
        if self.config.validation_metadata_file:
            splits.append(
                SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "metafile_path": self.config.validation_metadata_file,
                        "split": "validation",
                    },
                )
            )
        if self.config.test_metadata_file:
            splits.append(
                SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "metafile_path": self.config.test_metadata_file,
                        "split": "test",
                    },
                )
            )
        return splits

    def _generate_examples(self, metafile_path: str, split: str):
        # Load the metadata CSV/TSV
        dataset = load_dataset(
            "csv",
            data_files=[str(metafile_path)],
            split="train",
            delimiter="\t",
            num_proc=get_num_proc(),
        )

        # Filter missing files
        dataset = dataset.filter(lambda ex: file_exists_filter("signal", ex), num_proc=get_num_proc())

        def mapping_function(sample: Dict[str, Any]) -> Dict[str, Any]:
            video_path = sample["signal"]
            # Convert millisecond timestamps to seconds
            start_ms = sample.get("signal_start", 0) or 0
            end_ms   = sample.get("signal_end",   0) or 0
            start_sec = start_ms / 1000.0
            end_sec   = end_ms   / 1000.0 if end_ms > 0 else None

            # Try to open and seek; skip if no video stream or any error
            container = av.open(str(video_path))
            if not container.streams.video:
                sample["_invalid"] = True
                sample["DURATION"] = 0
                return sample

            stream = container.streams.video[0]
            start_pts = int(start_sec / float(stream.time_base))
            container.seek(start_pts, any_frame=False, backward=True, stream=stream)

            count_new = 0
            for frame in container.decode(video=0):
                timestamp = float(frame.pts * stream.time_base)
                if timestamp < start_sec:
                    continue
                if end_sec is not None and timestamp > end_sec:
                    break
                count_new += 1
            container.close()

            # If within ±2 frames of the threshold, fallback to old method
            maxf = self.max_frames
            if maxf is not None and abs(count_new - maxf) <= 2:
                # --- Fallback (torchvision.read_video) ---
                frames, _, _ = read_video(
                    str(video_path),
                    start_pts=start_sec,
                    end_pts=end_sec,
                    pts_unit="sec",
                )
                sample["DURATION"] = frames.shape[0]
            else:
                sample["DURATION"] = count_new

            sample["_invalid"] = False
            return sample

        # Map to extract duration
        dataset = dataset.map(mapping_function, num_proc=get_num_proc())
        dataset = dataset.filter(lambda ex: not ex.get("_invalid", False), num_proc=get_num_proc())

        # Filter by max_frames if set
        if self.max_frames is not None:
            dataset = dataset.filter(lambda ex: duration_filter(self.max_frames, ex), num_proc=get_num_proc())

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item["signal"],
                "signal_start": item.get("signal_start", 0),
                "signal_end": item.get("signal_end", 0),
                "encoder_prompt": item.get("encoder_prompt", "") or "",
                "decoder_prompt": item.get("decoder_prompt", "") or "",
                "output": item.get("output", ""),
            }