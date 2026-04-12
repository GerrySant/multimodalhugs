import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from torchvision.io import read_video
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

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
        io_max_retries: int = 3,
        signal_start_end_unit: str = "milliseconds",
    ):
        """
        Args:
            custom_preprocessor_path: HuggingFace model ID or local path to an
                image preprocessor (e.g. ``"openai/clip-vit-base-patch32"``).
                When provided, frames are decoded with OpenCV and passed through
                the preprocessor (e.g. CLIPImageProcessor). When None, frames
                are read with torchvision and returned as raw float tensors.
            skip_frames_stride: If set, keeps only every N-th frame along the
                temporal axis after loading (e.g. 2 → halve frame rate).
                None disables downsampling. Default: None.
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
                values in the dataset.  Either ``"milliseconds"`` (default,
                current behaviour — for the OpenCV path values are used with
                ``CAP_PROP_POS_MSEC``; for the torchvision path values are
                converted to seconds) or ``"frames"`` (values are used as
                frame indices: ``CAP_PROP_POS_FRAMES`` for the OpenCV path,
                direct tensor slicing for the torchvision path).
                When ``signal_start=0`` and ``signal_end=0`` the full file is
                always loaded regardless of this setting.
        """
        if custom_preprocessor_path is not None and not _CV2_AVAILABLE:
            raise ImportError(
                "VideoModalityProcessor with a custom_preprocessor_path requires 'opencv-python'. "
                'Install it with: pip install opencv-python  or  pip install "multimodalhugs[video]"'
            )
        if custom_preprocessor_path is None and not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "VideoModalityProcessor requires 'torchvision'. "
                'Install it with: pip install torchvision  or  pip install "multimodalhugs[video]"'
            )
        _valid_units = {"milliseconds", "frames"}
        if signal_start_end_unit not in _valid_units:
            raise ValueError(
                f"Invalid signal_start_end_unit '{signal_start_end_unit}'. "
                f"Must be one of: {sorted(_valid_units)}."
            )
        self.custom_preprocessor_path = custom_preprocessor_path
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

        When ``custom_preprocessor`` is set, frames are decoded with OpenCV
        and passed through the image preprocessor (e.g. CLIPImageProcessor).
        Otherwise, torchvision's ``read_video`` is used and frames are returned
        as raw float tensors. Retries up to ``io_max_retries`` times with
        exponential backoff on transient I/O failures.

        Args:
            video_path: Path to a video file (any format supported by OpenCV
                or torchvision).
            signal_start: Clip start value. When ``signal_start_end_unit`` is
                ``"milliseconds"`` this is a time in ms; when ``"frames"`` it
                is a frame index. 0.0 means start of file in both units.
            signal_end: Clip end value. Same unit logic as ``signal_start``.
                0.0 means end of file in both units.

        Returns:
            Float tensor of shape [T, C, H, W] (or [T, C*H*W] when
            ``join_chw=True``) where T is the number of frames after optional
            downsampling.

        Raises:
            IOError: If the video cannot be opened after ``io_max_retries``
                attempts.
        """
        last_exc: Exception = IOError(f"Cannot open video {video_path}")
        for attempt in range(max(1, self.io_max_retries)):
            try:
                if self.custom_preprocessor is not None:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        cap.release()
                        raise IOError(f"Cannot open video {video_path}")

                    full_signal = (signal_start == signal_end) or (signal_start is None) or (signal_end is None)
                    if not full_signal:
                        if self.signal_start_end_unit == "frames":
                            cap.set(cv2.CAP_PROP_POS_FRAMES, signal_start)
                        else:
                            cap.set(cv2.CAP_PROP_POS_MSEC, signal_start)

                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if not full_signal:
                            if self.signal_start_end_unit == "frames":
                                if cap.get(cv2.CAP_PROP_POS_FRAMES) > signal_end:
                                    break
                            else:
                                if cap.get(cv2.CAP_PROP_POS_MSEC) > signal_end:
                                    break
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    cap.release()

                    result = self.custom_preprocessor(images=frames, return_tensors="pt")["pixel_values"]
                    result = result.squeeze(0) if result.ndim == 5 else result
                else:
                    if self.signal_start_end_unit == "frames":
                        # torchvision's read_video does not support frame-index
                        # seeking, so we load the full video and slice by index.
                        result, _, _ = read_video(
                            str(video_path),
                            pts_unit="sec",
                            output_format="TCHW",
                        )
                        start_frame = int(signal_start) if signal_start else None
                        end_frame = int(signal_end) if signal_end else None
                        result = result[start_frame:end_frame].to(torch.float32)
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
