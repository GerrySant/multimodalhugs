import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from multimodalhugs.data import pad_and_create_mask, get_images, string_to_list
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput

logger = logging.getLogger(__name__)


class ImageModalityProcessor(ModalityProcessor):
    """
    Loads and preprocesses image sequences.

    Accepted inputs:
      - A file path (.npy, .jpg, .jpeg, .png, .bmp, .tiff, .tif) → loaded from disk.
      - A plain text string (no matching file) → rendered as a typographic image via
        ``get_images``.
      - A numpy array → converted to tensor directly.
      - A pre-loaded torch.Tensor → returned unchanged.
      - A pyarrow.lib.StringScalar → unwrapped to str and handled as above.

    process_sample — converts one signal value to a tensor.
    process_batch  — pads a list of tensors along the first dimension and returns a mask.
    """

    def __init__(
        self,
        font_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        normalize_image: bool = True,
        mean: Optional[Union[str, List[float]]] = None,
        std: Optional[Union[str, List[float]]] = None,
    ):
        """
        Args:
            font_path: Path to a TrueType font file used when rendering plain
                text strings as typographic images. Only required when the
                input signal is a text string with no matching file on disk.
                Default: None.
            width: Target width in pixels for text-rendered images. Ignored
                when loading from file. Default: None (uses get_images default).
            height: Target height in pixels for text-rendered images. Ignored
                when loading from file. Default: None (uses get_images default).
            normalize_image: If True, normalises pixel values using ``mean``
                and ``std``. Requires both to be provided. Default: True.
            mean: Per-channel mean for normalisation, as a list of floats or a
                comma-separated string (e.g. ``"0.485,0.456,0.406"``).
                Required when ``normalize_image=True``.
            std: Per-channel standard deviation for normalisation, as a list of
                floats or a comma-separated string.
                Required when ``normalize_image=True``.
        """
        if normalize_image and (mean is None or std is None):
            raise ValueError(
                "Normalization is enabled (normalize_image=True), but 'mean' and/or 'std' "
                "were not provided."
            )
        if isinstance(mean, str):
            mean = string_to_list(mean)
        if isinstance(std, str):
            std = string_to_list(std)
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
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".npy":
            image = np.load(path)
        elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        if image is None:
            raise ValueError(f"Failed to read image from path: {path}")
        if self.mean is not None and image.ndim >= 3 and len(self.mean) != image.shape[-1]:
            raise ValueError(
                f"Image at '{path}' has {image.shape[-1]} channels but "
                f"mean/std have {len(self.mean)} values."
            )
        if self.normalize_image:
            image = (image - self.mean) / self.std
        return torch.from_numpy(image)

    def _render_text(self, text: str) -> torch.Tensor:
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
            if os.path.exists(values):
                return self._load_from_path(values)
            return self._render_text(values)

        raise TypeError(f"Unsupported type for image input: {type(values)}")

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """Pad tensors along the first dimension and return a mask."""
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
