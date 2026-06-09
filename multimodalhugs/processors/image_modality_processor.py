import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import load_image

from multimodalhugs.data import pad_and_create_mask, get_images, string_to_list
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput

logger = logging.getLogger(__name__)


class ImageModalityProcessor(ModalityProcessor):
    """
    Loads and preprocesses image sequences.

    Accepted inputs:
      - A file path (.jpg, .jpeg, .png, .bmp, .tiff, .tif) → loaded via
        ``transformers.image_utils.load_image()``, which handles local paths,
        HTTP/HTTPS URLs, and base64-encoded strings. Images are always returned
        in RGB with EXIF rotation applied.
      - A file path (.npy) → loaded as a precomputed feature array via numpy.
      - A plain text string (no matching file) → rendered as a typographic image
        via ``get_images``.
      - A numpy array → converted to tensor directly.
      - A pre-loaded torch.Tensor → returned unchanged.
      - A pyarrow.lib.StringScalar → unwrapped to str and handled as above.

    **Output format** — ``process_sample`` always returns a float32 tensor of
    shape ``[T, C, H, W]`` where T is the number of frames/images in the sample:

    - Image file path (single image): T = 1, shape ``[1, C, H, W]``.
      Without ``custom_preprocessor_path``: pixel values in the 0–255 range
      (or normalised if ``normalize_image=True``).
      With ``custom_preprocessor_path``: resized and normalised by the
      specified HuggingFace image processor (e.g. ``CLIPImageProcessor``).
    - Text string (rendered as word images): T = number of words, each word
      rendered as a separate ``[C, H, W]`` image, shape ``[N_words, C, H, W]``.
    - ``.npy`` / pre-loaded numpy array / torch.Tensor: passed through unchanged
      (shape is the caller's responsibility).

    ``process_batch`` always calls ``pad_and_create_mask``, producing
    ``[B, T_max, C, H, W]`` with a ``[B, T_max]`` padding mask.  For single-image
    samples (T = 1) from files the output is ``[B, 1, C, H, W]``, which is
    compatible with a CLIP ``FeatureExtractor`` in the model.  For text-rendered
    sequences (T = N_words) the output is ``[B, N_max, C, H, W]``.

    process_sample — converts one signal value to a tensor.
    process_batch  — pads (or stacks) a list of tensors and returns a mask.
    """

    def __init__(
        self,
        custom_preprocessor_path: Optional[str] = None,
        font_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        normalize_image: bool = True,
        mean: Optional[Union[str, List[float]]] = None,
        std: Optional[Union[str, List[float]]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            custom_preprocessor_path: HuggingFace model ID or local path to an
                image processor (e.g. ``"openai/clip-vit-base-patch32"``). When
                provided, loaded images are passed through the processor, which
                handles resize, normalisation, and channel reordering. The output
                tensor is ``[C, H, W]`` float32. When ``None`` (default), images
                are returned as-is (with optional manual normalisation via
                ``normalize_image`` / ``mean`` / ``std``). Setting
                ``custom_preprocessor_path`` takes precedence: ``normalize_image``,
                ``mean``, and ``std`` are ignored when a preprocessor is set.
            font_path: Path to a TrueType font file used when rendering plain
                text strings as typographic images. Only required when the input
                signal is a text string with no matching file on disk.
                Default: None.
            width: Target width in pixels for text-rendered images. Ignored
                when loading from file. Default: None (uses get_images default).
            height: Target height in pixels for text-rendered images. Ignored
                when loading from file. Default: None (uses get_images default).
            normalize_image: If True, normalises pixel values using ``mean``
                and ``std``. Requires both to be provided. Ignored when
                ``custom_preprocessor_path`` is set. Default: True.
            mean: Per-channel mean for normalisation, as a list of floats or a
                comma-separated string (e.g. ``"0.485,0.456,0.406"``).
                Required when ``normalize_image=True`` and no preprocessor is set.
            std: Per-channel standard deviation for normalisation, as a list of
                floats or a comma-separated string.
                Required when ``normalize_image=True`` and no preprocessor is set.
            device: Device on which the ``custom_preprocessor`` runs its resize
                and normalisation steps (e.g. ``"cuda"`` or ``"cuda:0"``).
                In transformers 5.x the default ``TorchvisionBackend`` honours
                this argument and keeps all tensor operations on the specified
                device, enabling GPU-accelerated image preprocessing without any
                explicit ``.to(device)`` call in user code.  When ``None``
                (default) the preprocessor runs on CPU.  Ignored when
                ``custom_preprocessor_path`` is not set.
        """
        if custom_preprocessor_path is None and normalize_image and (mean is None or std is None):
            raise ValueError(
                "Normalization is enabled (normalize_image=True), but 'mean' and/or 'std' "
                "were not provided."
            )
        if custom_preprocessor_path is not None and normalize_image:
            logger.warning(
                "Both 'custom_preprocessor_path' and 'normalize_image=True' are set. "
                "The custom preprocessor handles normalisation internally; "
                "'normalize_image', 'mean', and 'std' will be ignored."
            )
        if isinstance(mean, str):
            mean = string_to_list(mean)
        if isinstance(std, str):
            std = string_to_list(std)

        self.custom_preprocessor_path = custom_preprocessor_path
        self.custom_preprocessor = (
            AutoProcessor.from_pretrained(custom_preprocessor_path)
            if custom_preprocessor_path is not None
            else None
        )
        self.font_path = font_path
        self.width = width
        self.height = height
        self.normalize_image = normalize_image
        self.mean = mean
        self.std = std
        self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_path(self, path: str) -> torch.Tensor:
        """
        Load an image from disk and optionally normalise or preprocess it.

        Args:
            path: Path to an image file. Supported formats:
                ``.npy`` (numpy array), ``.jpg``, ``.jpeg``, ``.png``,
                ``.bmp``, ``.tiff``, ``.tif`` (loaded via
                ``transformers.image_utils.load_image``; also accepts URLs).

        Returns:
            Float32 tensor of shape ``[1, C, H, W]``:
            - Without ``custom_preprocessor_path``: pixel values 0–255 (or
              normalised if ``normalize_image=True``).
            - With ``custom_preprocessor_path``: resized and normalised by the
              image processor.
            For ``.npy`` files: shape depends on the stored array (no T=1 wrap).

        Raises:
            ValueError: If the file extension is unsupported or the file cannot
                be read.
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".npy":
            image = np.array(np.load(path), dtype=np.float32)
            if self.normalize_image and self.custom_preprocessor is None:
                if self.mean is not None and image.ndim >= 3 and len(self.mean) != image.shape[-1]:
                    raise ValueError(
                        f"Image at '{path}' has {image.shape[-1]} channels but "
                        f"mean/std have {len(self.mean)} values."
                    )
                image = (image - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
            return torch.from_numpy(image)
        elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
            pil_image = load_image(path)  # PIL RGB, EXIF rotation applied
            if self.custom_preprocessor is not None:
                # [1, C, H, W] — keep the leading dim so process_batch sees [T, C, H, W]
                call_kwargs = {"images": pil_image, "return_tensors": "pt"}
                if self.device is not None:
                    call_kwargs["device"] = self.device
                return self.custom_preprocessor(**call_kwargs)["pixel_values"]
            image = np.array(pil_image, dtype=np.float32)  # [H, W, 3] RGB
            if self.normalize_image:
                if self.mean is not None and len(self.mean) != image.shape[-1]:
                    raise ValueError(
                        f"Image at '{path}' has {image.shape[-1]} channels but "
                        f"mean/std have {len(self.mean)} values."
                    )
                image = (image - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
            # [H, W, C] → [C, H, W] → [1, C, H, W] so process_batch sees [T, C, H, W]
            return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _render_text(self, text: str) -> torch.Tensor:
        """
        Render a plain text string as a typographic image sequence.

        Each word in ``text`` is rendered as a separate ``[C, H, W]`` image.
        When ``custom_preprocessor`` is set, raw (un-normalised) frames are
        returned so the preprocessor can apply its own normalisation; manual
        normalisation via ``normalize_image`` / ``mean`` / ``std`` is skipped.

        Args:
            text: The text string to render using the configured font.

        Returns:
            Float tensor of shape ``[N_words, C, H, W]``.
            Pixel values are in the 0–255 range when ``custom_preprocessor``
            is set; otherwise optionally normalised by ``mean`` / ``std``.
        """
        apply_normalize = self.normalize_image and self.custom_preprocessor is None
        image = get_images(
            src_text=text,
            font_path=self.font_path,
            width=self.width,
            height=self.height,
            normalize_image=apply_normalize,
            mean=self.mean,
            std=self.std,
        )
        return torch.from_numpy(image)

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Load and preprocess a single image sample. Called at dataset-transform time.

        Args:
            values: One of:
                - torch.Tensor — returned unchanged.
                - np.ndarray — converted to a tensor unchanged.
                - str (existing file path or URL) — loaded via
                  ``_load_from_path``. URLs (http/https) are supported for
                  standard image formats.
                - str (no matching file, not a URL) — rendered as a typographic
                  image via ``_render_text``.
                - pyarrow.lib.StringScalar — unwrapped to str and handled as
                  above.

        Returns:
            Float32 tensor of shape ``[T, C, H, W]``:
            - Image file / URL: T=1 → ``[1, C, H, W]``.
            - Text string: T=N_words → ``[N_words, C, H, W]``.
            ``custom_preprocessor`` is applied to image files and rendered text
            frames; the output size is determined by the preprocessor.
            ``.npy`` files, numpy arrays, and tensors are passed through
            unchanged (precomputed features — ``custom_preprocessor`` is
            not applied).

        Raises:
            TypeError: If ``values`` is of an unsupported type.
        """
        if isinstance(values, torch.Tensor):
            return values
        if isinstance(values, np.ndarray):
            return torch.from_numpy(values)

        # Unwrap pyarrow scalar if needed
        try:
            import pyarrow
            if isinstance(values, pyarrow.lib.StringScalar):
                values = values.as_py()
        except ImportError:
            pass

        if isinstance(values, str):
            is_url = values.startswith("http://") or values.startswith("https://")
            if is_url or os.path.exists(values):
                return self._load_from_path(values)
            # Text rendering — apply custom_preprocessor if set
            frames = self._render_text(values)  # [N_words, C, H, W], values 0-255
            if self.custom_preprocessor is not None:
                # Convert to list of PIL images (uint8 RGB) for the preprocessor
                frames_np = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                pil_frames = [Image.fromarray(f) for f in frames_np]
                call_kwargs = {"images": pil_frames, "return_tensors": "pt"}
                if self.device is not None:
                    call_kwargs["device"] = self.device
                result = self.custom_preprocessor(**call_kwargs)["pixel_values"]  # [N_words, C', H', W']
                return result
            return frames

        raise TypeError(f"Unsupported type for image input: {type(values)}")

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """
        Batch a list of image tensors. Called at collation time.

        Pads a list of ``[T_i, C, H, W]`` tensors along the T dimension,
        producing ``[B, T_max, C, H, W]`` with a ``[B, T_max]`` mask.

        For single image files T = 1, so output is ``[B, 1, C, H, W]`` —
        compatible with a CLIP ``FeatureExtractor`` in the model.
        For text-rendered sequences T = N_words, so output is
        ``[B, N_max, C, H, W]``.

        Args:
            samples: List of B tensors as returned by ``process_sample``.

        Returns:
            ProcessBatchOutput(data, mask).
        """
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
