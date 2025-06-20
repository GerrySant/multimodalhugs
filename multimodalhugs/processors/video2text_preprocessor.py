# Standard library
import logging
import os

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

# Third-party libraries
import cv2
import psutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.io import read_video
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

# Local application imports
from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor
from multimodalhugs.processors.utils import frame_skipping


logger = logging.getLogger(__name__)

def get_dynamic_cache_size():
    cluster_mem = None
    for var in ("SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        if os.getenv(var):
            cluster_mem = int(os.getenv(var)) * 1e6
            break
    total = cluster_mem or psutil.virtual_memory().total
    size = int((total * 0.05) / 50e6)  # 5% of RAM, ~50MB per video
    return max(10, size)

class Video2TextTranslationProcessor(MultimodalSequence2SequenceProcessor):
    """
    Reads video (path / ndarray / tensor) → [T, C, H, W] float32, optionally normalizes,
    resizes, then pads & masks.
    """
    name = "video2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["inputs_embeds", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        custom_preprocessor_path: Optional[str] = None,
        skip_frames_stride: Optional[int] = None, 
        join_chw: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Initializes the Video2TextTranslationProcessor.

        Args:
            tokenizer (Optional[Any]): A Hugging Face tokenizer used for processing the text output.
            custom_preprocessor_path (Optional[str]): Optional path to a custom `AutoProcessor` to use for video preprocessing.
            skip_frames_stride (Optional[int]):  If set, skips input frames at the given stride.
                Useful for downsampling frame sequences during preprocessing.
            join_chw (bool): If True, flattens each video frame from [C, H, W] to [C*H*W] before feeding to the model.
            use_cache (bool): If True, enables an LRU cache to store preprocessed videos for faster repeated access.
            **kwargs: Additional keyword arguments passed to the parent class `MultimodalSequence2SequenceProcessor`.
        """
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.custom_preprocessor_path = custom_preprocessor_path
        self.join_chw = join_chw
        self.skip_frames_stride = skip_frames_stride
        self.use_cache = use_cache
        self.custom_preprocessor = AutoProcessor.from_pretrained(self.custom_preprocessor_path) if self.custom_preprocessor_path is not None else None

        if self.use_cache:
            cache_size = get_dynamic_cache_size()
            logger.info(f" Video cache size: {cache_size}")
            self._video_file_to_tensor = lru_cache(maxsize=cache_size)(
                self._video_file_to_tensor
            )

    def _video_file_to_tensor(
        self,
        video_input: Union[str, Path, np.ndarray, torch.Tensor],
        signal_start: float = 0.0,
        signal_end: float = 0.0
    ) -> torch.Tensor:
        # passthrough tensor
        if isinstance(video_input, torch.Tensor):
            return video_input

        # ndarray → tensor
        if isinstance(video_input, np.ndarray):
            return torch.from_numpy(video_input)

        if self.custom_preprocessor:

            cap = cv2.VideoCapture(video_input)
            if not cap.isOpened():
                raise IOError(f"Cannot open video {video_input}")

            full_sigal = (signal_start == signal_end) or (signal_start is None) or (signal_end is None)

            if not full_sigal:
                # jump to start time
                cap.set(cv2.CAP_PROP_POS_MSEC, signal_start)

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                curr_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if (curr_ms > signal_end) and not full_sigal:
                    break

                # BGR → RGB → PIL
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(img)
            cap.release()

            # process entire interval in one batch
            frames = self.custom_preprocessor(images=frames, return_tensors="pt")["pixel_values"]
            frames = frames.squeeze(0) if frames.ndim == 5 else frames # If [1, T, C, H, W] -> [T, C, H, W]
            frames = frame_skipping(x=frames, t_dim=0, stride=self.skip_frames_stride) if self.skip_frames_stride is not None else frames
            return frames

        # read from disk (convert ms → sec)
        start_sec = (signal_start or 0) / 1000.0
        end_sec   = (signal_end / 1000.0) if signal_end else None
        frames, _, _ = read_video(
            str(video_input),
            start_pts=start_sec,
            end_pts=end_sec,
            pts_unit="sec",
            output_format="TCHW"
        )
        frames = frame_skipping(x=frames, t_dim=0, stride=self.skip_frames_stride) if self.skip_frames_stride is not None else frames
        return frames.to(torch.float32) # [T, C, H, W]

    def _obtain_multimodal_input_and_masks(self, batch: BatchFeature, **kwargs):
        sequences = [
            self._video_file_to_tensor(
                sample["signal"],
                sample.get("signal_start", 0.0),
                sample.get("signal_end", 0.0),
            )
            for sample in batch
        ]
        padded, masks = pad_and_create_mask(sequences)
        if self.join_chw: 
            B, T = padded.shape[:2]
            padded = padded.view(B, T, -1)         # [B, T, C, H, W] → [B, T, C*H*W]
        return {
            "input_frames": padded, # [B, T, C, H, W] when self.join_chw set to False. [B, T, *] when self.join_chw set to True.
            "attention_mask": masks # [B, T]
            }, kwargs

    def _transform_get_items_output(self, batch):
        """
        Returns a transformation function applied at the dataset level during iteration.

        This method defines a transformation that is applied to each batch **within the dataset iterator**, 
        typically by using `datasets.Dataset.with_transform()`. As a result, the transformation is executed 
        at runtime during `__getitem__()` or `__getitems__()`, which allows it to benefit from prefetching 
        and parallel data loading when using multiple DataLoader workers.

        Unlike the `_obtain_*` methods, which are also executed on-the-fly but within the **processor call 
        (typically inside the DataCollator)**, this transformation occurs **prior to batching and collation**. 
        It is therefore ideal for operations that are expensive and can be parallelized at the sample or batch 
        level, such as decoding signals, loading external files, or converting inputs to intermediate formats.

        Use this method to preprocess inputs early in the pipeline while maintaining a modular design that 
        separates dataset-level and collator-level responsibilities.

        Args:
            batch (Dict[str, List[Any]]): A dictionary representing a batch of dataset examples (not yet collated).

        Returns:
            Dict[str, List[Any]]: The transformed batch, with updated or added fields ready for collation.
        """
        tensor_signals = [self._video_file_to_tensor(batch["signal"][i], batch["signal_start"][i], batch["signal_end"][i]) for i in range(len(batch["signal"]))]
        batch["signal"] = tensor_signals
        return batch