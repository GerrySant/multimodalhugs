import logging
import multiprocessing
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.video_utils import load_video, VideoMetadata

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput
from multimodalhugs.processors.utils import get_dynamic_cache_size, SignalUnit

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = {"pyav", "torchvision", "torchcodec", "decord", "opencv"}


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
        backend: str = "pyav",
        num_frames: Optional[int] = None,
        skip_frames_stride: Optional[int] = None,
        join_chw: bool = False,
        use_cache: bool = False,
        io_max_retries: int = 3,
        signal_start_end_unit: SignalUnit = SignalUnit.MILLISECONDS,
    ):
        """
        Args:
            custom_preprocessor_path: HuggingFace model ID or local path to an
                image preprocessor (e.g. ``"openai/clip-vit-base-patch32"``).
                When provided, frames are passed through the preprocessor
                (e.g. CLIPImageProcessor) after decoding. When None, frames are
                returned as raw float tensors.
            backend: Video decoding backend. One of ``"pyav"`` (default),
                ``"torchvision"``, ``"torchcodec"``, ``"decord"``, or
                ``"opencv"``. The chosen backend must be installed; if it is not,
                ``load_video`` raises a clear ``ImportError`` at call time.
                ``"torchcodec"`` decodes directly to GPU tensors and eliminates
                the CPU→GPU copy bottleneck for large-batch training.
            num_frames: If set, uniformly subsample this many frames from the
                clip window. Takes precedence over ``skip_frames_stride`` when
                both are set. Default: None (keep all frames).
            skip_frames_stride: If set, keeps only every N-th frame along the
                temporal axis after loading (e.g. 2 → halve frame rate).
                Ignored when ``num_frames`` is set. Default: None.
            join_chw: If True, merges the channel, height, and width dimensions
                into a single feature dimension, producing shape [B, T, C*H*W]
                instead of [B, T, C, H, W]. Default: False.
            use_cache: If True, wraps ``_load_video`` with an LRU cache whose
                size is derived from available system (or SLURM) memory,
                assuming ~50 MB per cached video. Useful when the same video
                clips are repeated across epochs. Default: False.
            io_max_retries: Maximum number of attempts when opening a video
                file fails with an I/O error. Retries use exponential backoff
                (1 s, 2 s, 4 s, …) to tolerate transient NFS slowness on
                shared clusters. Default: 3.
            signal_start_end_unit: Unit for ``signal_start`` / ``signal_end``
                values in the dataset. Either ``SignalUnit.MILLISECONDS``
                (default) or ``SignalUnit.FRAMES`` (values are used as frame
                indices). When ``signal_start=0`` and ``signal_end=0`` the full
                file is always loaded regardless of this setting.
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {SUPPORTED_BACKENDS}, got '{backend}'"
            )
        if backend == "torchcodec":
            start_method = multiprocessing.get_start_method(allow_none=True)
            if start_method is None:
                start_method = "fork" if sys.platform.startswith("linux") else "spawn"
            if start_method == "fork":
                logger.warning(
                    "backend='torchcodec' decodes video on CUDA. The current (or default) "
                    "multiprocessing start method is 'fork', which cannot safely inherit a "
                    "CUDA context into DataLoader worker processes — this will cause CUDA "
                    "errors or deadlocks when num_workers > 0. "
                    "Call torch.multiprocessing.set_start_method('spawn') before training, "
                    "or set dataloader_num_workers=0."
                )
        try:
            signal_start_end_unit = SignalUnit(signal_start_end_unit)
        except ValueError:
            raise ValueError(
                f"Invalid signal_start_end_unit '{signal_start_end_unit}'. "
                f"Must be one of: {[u.value for u in SignalUnit]}."
            )
        self.custom_preprocessor_path = custom_preprocessor_path
        self.backend = backend
        self.num_frames = num_frames
        self.skip_frames_stride = skip_frames_stride
        self.join_chw = join_chw
        self.use_cache = use_cache
        self.io_max_retries = io_max_retries
        self.signal_start_end_unit = signal_start_end_unit
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
        """
        Load a video clip from disk and apply the preprocessing pipeline.
        Retries up to ``io_max_retries`` times with exponential backoff on
        transient I/O failures.

        Args:
            video_path: Path to a video file.
            signal_start: Clip start value in the unit given by
                ``signal_start_end_unit``. 0.0 means start of file.
            signal_end: Clip end value. 0.0 means end of file.

        Returns:
            Float tensor of shape [T, C, H, W].

        Raises:
            IOError: If the video cannot be opened after ``io_max_retries``
                attempts.
        """
        last_exc: Exception = IOError(f"Cannot open video {video_path}")
        for attempt in range(max(1, self.io_max_retries)):
            try:
                return self._load_video_impl(video_path, signal_start, signal_end)
            except (IOError, OSError, RuntimeError) as exc:
                last_exc = exc
                if attempt < self.io_max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "Failed to load video '%s' (attempt %d/%d): %s. Retrying in %ds.",
                        video_path, attempt + 1, self.io_max_retries, exc, delay,
                    )
                    time.sleep(delay)

        raise IOError(
            f"Cannot open video '{video_path}' after {self.io_max_retries} attempts."
        ) from last_exc

    def _load_video_impl(
        self,
        video_path: Union[str, Path],
        signal_start: float = 0.0,
        signal_end: float = 0.0,
    ) -> torch.Tensor:
        """Actual video loading via ``transformers.video_utils.load_video``."""

        def _sample_indices_fn(metadata: VideoMetadata, **kwargs):
            total = metadata.total_num_frames
            if signal_start == signal_end == 0:
                start_frame, end_frame = 0, total
            elif self.signal_start_end_unit == SignalUnit.FRAMES:
                start_frame = int(signal_start) if signal_start else 0
                end_frame = int(signal_end) if signal_end else total
            else:  # MILLISECONDS
                fps = metadata.fps or 25.0
                start_frame = int((signal_start / 1000.0) * fps)
                end_frame = int((signal_end / 1000.0) * fps) if signal_end else total
                end_frame = min(end_frame, total)

            clip_frames = end_frame - start_frame
            if self.num_frames is not None and self.num_frames < clip_frames:
                indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
            elif self.skip_frames_stride is not None:
                indices = np.arange(start_frame, end_frame, self.skip_frames_stride)
            else:
                indices = np.arange(start_frame, end_frame)

            return indices

        frames, _ = load_video(
            str(video_path),
            backend=self.backend,
            sample_indices_fn=_sample_indices_fn,
        )
        # frames: np.ndarray [T, H, W, C] uint8 for most backends;
        #         torch.Tensor [T, C, H, W] for torchcodec (GPU-native)

        if self.custom_preprocessor is not None:
            if isinstance(frames, torch.Tensor):
                # torchcodec returns a tensor; move to CPU for preprocessors that
                # expect numpy/PIL input
                result = self.custom_preprocessor(
                    images=frames.cpu(), return_tensors="pt"
                )["pixel_values"]
            else:
                pil_frames = [Image.fromarray(f) for f in frames]
                result = self.custom_preprocessor(
                    images=pil_frames, return_tensors="pt"
                )["pixel_values"]
            result = result.squeeze(0) if result.ndim == 5 else result
        else:
            if isinstance(frames, torch.Tensor):
                # torchcodec already returns [T, C, H, W]
                result = frames.to(torch.float32)
            else:
                # THWC uint8 → TCHW float32
                result = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32)

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
        Load and preprocess a single video sample. Called at dataset-transform time.

        Args:
            values: One of:
                - str or Path — path to a video file; loaded and preprocessed.
                - torch.Tensor — returned unchanged (already preprocessed).
                - np.ndarray — converted to a float tensor unchanged.
                - dict — mapping with keys:
                    ``"signal"`` (str/Path, required): path to the video file.
                    ``"signal_start"`` (float, optional): clip start in the unit
                    given by ``signal_start_end_unit``. Default 0.0 (start of file).
                    ``"signal_end"`` (float, optional): clip end in the unit given
                    by ``signal_start_end_unit``. Default 0.0 (end of file).

        Returns:
            Float tensor of shape [T, C, H, W].
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
        Pad a batch of video tensors to a common length. Called at collation time.

        Args:
            samples: List of B tensors, each of shape [T_i, C, H, W] (or
                [T_i, C*H*W] when ``join_chw=True``), as returned by
                ``process_sample``.

        Returns:
            ProcessBatchOutput where:
                - data: Float tensor of shape [B, T_max, C, H, W] (or
                  [B, T_max, C*H*W] when ``join_chw=True``), zero-padded.
                - mask: Bool tensor of shape [B, T_max], True for valid frames.
        """
        padded, mask = pad_and_create_mask(samples)
        if self.join_chw:
            B, T = padded.shape[:2]
            padded = padded.view(B, T, -1)
        return ProcessBatchOutput(data=padded, mask=mask)
