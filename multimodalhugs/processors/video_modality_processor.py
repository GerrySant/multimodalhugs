import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_video
from transformers import AutoProcessor

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput
from multimodalhugs.processors.utils import frame_skipping, get_dynamic_cache_size

logger = logging.getLogger(__name__)


class VideoModalityProcessor(ModalityProcessor):
    """
    Loads and preprocesses video sequences.

    process_sample — reads a video file, optionally applies a custom frame
                     preprocessor (e.g. CLIPImageProcessor), returns [T, C, H, W].
    process_batch  — pads a list of [T_i, ...] tensors to [B, T_max, ...] and
                     returns a [B, T_max] attention mask.
    """

    def __init__(
        self,
        custom_preprocessor_path: Optional[str] = None,
        skip_frames_stride: Optional[int] = None,
        join_chw: bool = False,
        use_cache: bool = False,
    ):
        self.custom_preprocessor_path = custom_preprocessor_path
        self.skip_frames_stride = skip_frames_stride
        self.join_chw = join_chw
        self.use_cache = use_cache
        self.custom_preprocessor = (
            AutoProcessor.from_pretrained(custom_preprocessor_path)
            if custom_preprocessor_path is not None
            else None
        )
        if use_cache:
            cache_size = get_dynamic_cache_size(avg_item_size_bytes=50e6)  # ~50 MB per video
            logger.info(f"Video cache size: {cache_size}")
            self._load_video = lru_cache(maxsize=cache_size)(self._load_video)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_video(
        self,
        video_path: Union[str, Path],
        signal_start: float = 0.0,
        signal_end: float = 0.0,
    ) -> torch.Tensor:
        if self.custom_preprocessor is not None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open video {video_path}")

            full_signal = (signal_start == signal_end) or (signal_start is None) or (signal_end is None)
            if not full_signal:
                cap.set(cv2.CAP_PROP_POS_MSEC, signal_start)

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if not full_signal and cap.get(cv2.CAP_PROP_POS_MSEC) > signal_end:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

            result = self.custom_preprocessor(images=frames, return_tensors="pt")["pixel_values"]
            result = result.squeeze(0) if result.ndim == 5 else result
        else:
            start_sec = (signal_start or 0) / 1000.0
            end_sec = (signal_end / 1000.0) if signal_end else None
            result, _, _ = read_video(
                str(video_path),
                start_pts=start_sec,
                end_pts=end_sec,
                pts_unit="sec",
                output_format="TCHW",
            )
            result = result.to(torch.float32)

        if self.skip_frames_stride is not None:
            result = frame_skipping(x=result, t_dim=0, stride=self.skip_frames_stride)
        return result

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        values — file path, ndarray, or pre-loaded tensor, or a dict:
                   {"signal": path, "signal_start": float, "signal_end": float}
        """
        if isinstance(values, dict):
            signal = values["signal"]
            signal_start = values.get("signal_start", 0.0)
            signal_end = values.get("signal_end", 0.0)
        else:
            signal = values
            signal_start = kwargs.get("signal_start", 0.0)
            signal_end = kwargs.get("signal_end", 0.0)

        if isinstance(signal, torch.Tensor):
            return signal
        if isinstance(signal, np.ndarray):
            return torch.from_numpy(signal)

        return self._load_video(signal, signal_start, signal_end)

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """
        Pad [T_i, ...] tensors to [B, T_max, ...] and return a [B, T_max] mask.
        """
        padded, mask = pad_and_create_mask(samples)
        if self.join_chw:
            B, T = padded.shape[:2]
            padded = padded.view(B, T, -1)
        return ProcessBatchOutput(data=padded, mask=mask)
