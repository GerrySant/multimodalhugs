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

    **Output format** depends on whether ``custom_preprocessor_path`` is set:

    *Without ``custom_preprocessor_path``* (default): ``process_sample`` returns a
    float32 tensor of shape ``[H, W, C]`` with pixel values in the 0–255 range
    (or normalised if ``normalize_image=True``). ``process_batch`` pads variable-
    height images along H, producing ``[B, H_max, W_max, C]``.

    *With ``custom_preprocessor_path``*: ``process_sample`` passes the loaded PIL
    image through the specified HuggingFace image processor (e.g.
    ``"openai/clip-vit-base-patch32"``), which handles resize, normalisation, and
    channel reordering, returning a ``[C, H, W]`` float32 tensor. Because the
    preprocessor produces a fixed output size, ``process_batch`` simply stacks
    samples to ``[B, C, H, W]`` without padding, and returns ``mask=None``.
    ``normalize_image``, ``mean``, and ``std`` are ignored when a custom
    preprocessor is set.

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
            - Without ``custom_preprocessor_path``: float32 tensor ``[H, W, C]``,
              pixel values 0–255 (or normalised if ``normalize_image=True``).
            - With ``custom_preprocessor_path``: float32 tensor ``[C, H, W]`` in
              the preprocessor's expected range.

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
                result = self.custom_preprocessor(images=pil_image, return_tensors="pt")["pixel_values"]
                return result.squeeze(0)  # [C, H, W]
            image = np.array(pil_image, dtype=np.float32)  # [H, W, 3] RGB
            if self.normalize_image:
                if self.mean is not None and len(self.mean) != image.shape[-1]:
                    raise ValueError(
                        f"Image at '{path}' has {image.shape[-1]} channels but "
                        f"mean/std have {len(self.mean)} values."
                    )
                image = (image - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
            return torch.from_numpy(image)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _render_text(self, text: str) -> torch.Tensor:
        """
        Render a plain text string as a typographic image.

        Args:
            text: The text string to render using the configured font.

        Returns:
            Float tensor of shape (N_words, C, H, W), optionally normalised.
        """
        image = get_images(
            src_text=text,
            font_path=self.font_path,
            width=self.width,
            height=self.height,
            normalize_image=self.normalize_image,
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
            - Without ``custom_preprocessor_path``: float32 tensor ``[H, W, C]``
              (or ``[H, W]`` for grayscale .npy), optionally normalised.
            - With ``custom_preprocessor_path``: float32 tensor ``[C, H, W]``
              in the preprocessor's expected range.

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
            return self._render_text(values)

        raise TypeError(f"Unsupported type for image input: {type(values)}")

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """
        Batch a list of image tensors. Called at collation time.

        Without ``custom_preprocessor_path``: pads variable-size images to a
        common shape. Returns ``[B, H_max, W_max, C]`` data and a ``[B, H_max]``
        mask indicating valid rows.

        With ``custom_preprocessor_path``: all images are the same size (the
        preprocessor resizes them). Stacks without padding to ``[B, C, H, W]``
        and returns ``mask=None``.

        Args:
            samples: List of B tensors as returned by ``process_sample``.

        Returns:
            ProcessBatchOutput(data, mask).
        """
        if self.custom_preprocessor is not None:
            data = torch.stack(samples, dim=0)  # [B, C, H, W]
            return ProcessBatchOutput(data=data, mask=None)
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
