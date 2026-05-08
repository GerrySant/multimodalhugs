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

    Video decoding is delegated to ``transformers.video_utils.load_video()``,
    which supports five backends: ``"pyav"`` (default), ``"torchvision"``,
    ``"decord"``, ``"opencv"``, and ``"torchcodec"``. The backend is selected
    via the ``backend`` constructor parameter and is independent of
    ``custom_preprocessor_path``.

    **torchcodec: CPU vs GPU decode**

    ``backend="torchcodec"`` uses hardware codec infrastructure for video
    decoding. By default (``device=None``) it decodes to a **CPU tensor** —
    faster than software decoders (pyav, cv2) but still on CPU.

    To get true GPU decode, set ``device="cuda"`` (or ``"cuda:N"`` for a
    specific device). This routes decoding through NVDEC — a dedicated
    hardware video decode engine on NVIDIA GPUs that runs independently of
    CUDA compute cores. Decoded frames arrive as CUDA tensors without any
    CPU→GPU copy, and NVDEC can run in parallel with the training forward/
    backward pass on the same GPU without competing for compute resources.

    **torchcodec + DataLoader workers (GPU decode only)**

    When ``device="cuda"``, DataLoader workers must be started with the
    ``"spawn"`` multiprocessing start method. On Linux, PyTorch defaults to
    ``"fork"``, which cannot safely inherit a CUDA context into child processes
    and will cause CUDA errors or deadlocks when ``num_workers > 0``.

    To use GPU decode safely with multiple workers::

        import torch
        torch.multiprocessing.set_start_method("spawn")
        # then create your DataLoader / Trainer as normal

    Or set ``worker_start_method: spawn`` in the training config (see
    ``ExtendedSeq2SeqTrainingArguments``), or ``dataloader_num_workers=0``.

    ``VideoModalityProcessor`` emits a ``logger.warning`` at construction time
    when ``device="cuda"`` is combined with the ``"fork"`` start method (or
    its Linux default).

    **torchcodec + custom_preprocessor_path (GPU decode)**

    With transformers 5.x, the default ``TorchvisionBackend`` image processor
    accepts tensors directly and runs on the same device as the input. When
    ``device="cuda"`` and ``custom_preprocessor_path`` is set, decoded CUDA
    tensors are passed directly to the preprocessor without any CPU transfer —
    the full zero-copy GPU pipeline (decode + preprocess on GPU) is available.

    This works because the ``device`` argument is forwarded to the preprocessor
    call, and ``TorchvisionBackend`` honours it when inputs are tensors. For
    CPU-decoded backends (``pyav``, ``opencv``, ``decord``), passing
    ``device="cuda"`` will move frames to GPU for the resize and normalize
    steps, saving compute at the cost of one CPU→GPU transfer.
    """

    def __init__(
        self,
        custom_preprocessor_path: Optional[str] = None,
        backend: str = "pyav",
        device: Optional[str] = None,
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
                ``"opencv"``. The chosen backend must be installed; if it is
                not, ``load_video`` raises a clear ``ImportError`` at call
                time.

                - ``"pyav"`` (default): CPU, widely available, handles most
                  formats. Safe with any number of DataLoader workers.
                - ``"torchvision"``: CPU, PyTorch-native.
                - ``"decord"``: CPU, fast random access.
                - ``"opencv"``: CPU, no audio support.
                - ``"torchcodec"``: Hardware codec decode. Defaults to CPU
                  unless ``device="cuda"`` is also set, in which case NVDEC
                  is used for GPU-native decode. See the ``device`` parameter.
            device: Decode device for the ``"torchcodec"`` backend. ``None``
                (default) uses torchcodec's default (CPU). Set to ``"cuda"``
                or ``"cuda:N"`` to decode directly to GPU tensors via NVDEC,
                eliminating the CPU→GPU copy. Ignored for all other backends
                (a ``logger.warning`` is emitted if set with a non-torchcodec
                backend). When set to a CUDA device, the ``"spawn"``
                multiprocessing start method is required for DataLoader
                workers with ``num_workers > 0`` on Linux (see class
                docstring and ``worker_start_method`` in training config).
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
        if backend == "torchcodec" and device is None:
            if torch.cuda.is_available():
                detected_device = f"cuda:{torch.cuda.current_device()}"
                device = detected_device
                logger.warning(
                    "backend='torchcodec' selected without an explicit device. "
                    "Automatically using device='%s' (detected from torch.cuda.current_device()). "
                    "To use a specific device or force CPU decode, set the device argument "
                    "explicitly (e.g. device='cuda:0' or device='cpu').",
                    detected_device,
                )
        if device is not None and backend != "torchcodec":
            logger.warning(
                "device='%s' is only used by the 'torchcodec' backend; "
                "it will be ignored for backend='%s'.",
                device, backend,
            )
        if device is not None and str(device).startswith("cuda"):
            start_method = multiprocessing.get_start_method(allow_none=True)
            if start_method is None:
                start_method = "fork" if sys.platform.startswith("linux") else "spawn"
            if start_method == "fork":
                logger.warning(
                    "device='%s' enables CUDA decode via torchcodec. The current (or default) "
                    "multiprocessing start method is 'fork', which cannot safely inherit a "
                    "CUDA context into DataLoader worker processes — this will cause CUDA "
                    "errors or deadlocks when num_workers > 0. "
                    "Set worker_start_method: spawn in your training config, call "
                    "torch.multiprocessing.set_start_method('spawn') before training, "
                    "or set dataloader_num_workers=0.",
                    device,
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
        self.device = device
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

        load_video_kwargs = {"backend": self.backend, "sample_indices_fn": _sample_indices_fn}
        if self.device is not None:
            load_video_kwargs["device"] = self.device
        frames, _ = load_video(str(video_path), **load_video_kwargs)
        # frames: np.ndarray [T, H, W, C] uint8  — pyav, opencv, decord
        #         torch.Tensor [T, C, H, W]       — torchvision (deprecated in transformers 5.5), torchcodec

        if self.custom_preprocessor is not None:
            if isinstance(frames, torch.Tensor):
                # torchvision / torchcodec: pass as list of [C, H, W] tensors.
                # TorchvisionBackend (transformers 5.x default) processes tensors natively
                # and keeps them on their original device — no CPU transfer for GPU frames.
                input_frames = list(frames)
            else:
                # numpy [T, H, W, C] uint8 → list of PIL
                input_frames = [Image.fromarray(f) for f in frames]
            call_kwargs = {"images": input_frames, "return_tensors": "pt"}
            if self.device is not None:
                call_kwargs["device"] = self.device
            result = self.custom_preprocessor(**call_kwargs)["pixel_values"]
            result = result.squeeze(0) if result.ndim == 5 else result
        else:
            if isinstance(frames, torch.Tensor):
                # torchvision / torchcodec: already [T, C, H, W]; keep on original device
                result = frames.to(torch.float32)
            else:
                # numpy [T, H, W, C] uint8 → [T, C, H, W] float32
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
