import os
import torch
import torch.nn.functional as F
import logging
import psutil
import numpy as np

from pathlib import Path
from functools import lru_cache
from typing import Union, Optional, Tuple, Any

from torchvision.io import read_video
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor

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

class Video2TextTranslationProcessor(MultimodalSecuence2TextTranslationProcessor):
    """
    Reads video (path / ndarray / tensor) → [T, C, H, W] float32, optionally normalizes,
    resizes, then pads & masks.
    """
    attributes = ["tokenizer"]
    model_input_names = ["inputs_embeds", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        normalize: bool = True,
        resize: Optional[Union[int, Tuple[int, int]]] = None,
        join_chw: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Args:
            tokenizer (Optional[Any]): HuggingFace tokenizer instance for text side.
            normalize (bool): If True, scale pixel values to [0,1].
            resize (int | tuple[int, int] | None):
                - If int, resize frames to (resize, resize).
                - If tuple (H, W), resize to that shape.
                - If None, keep original frame size.
            use_cache (bool): If True, cache decoded videos in an LRU cache.
            **kwargs: Passed to the parent MultimodalSecuence2TextTranslationProcessor.
        """
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.normalize = normalize
        self.resize = resize
        self.join_chw = join_chw
        self.use_cache = use_cache
        if self.use_cache:
            cache_size = get_dynamic_cache_size()
            logger.info(f"Video cache size: {cache_size}")
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

        # read from disk (convert ms → sec)
        start_sec = (signal_start or 0) / 1000.0
        end_sec   = (signal_end / 1000.0) if signal_end else None
        frames, _, _ = read_video(
            str(video_input),
            start_pts=start_sec,
            end_pts=end_sec,
            pts_unit="sec",
        )
        # [T, H, W, C] → [T, C, H, W], float32
        frames = frames.permute(0, 3, 1, 2).to(torch.float32)
        if self.normalize:
            frames = frames / 255.0
        if self.resize:
            # allow int or (H, W)
            size = (self.resize, self.resize) if isinstance(self.resize, int) else self.resize
            # treat T as batch dimension
            frames = F.interpolate(frames, size=size, mode='bilinear', align_corners=False)

        return frames # [T, C, H, W]

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