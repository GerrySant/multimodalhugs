import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput
from multimodalhugs.processors.utils import frame_skipping, get_dynamic_cache_size

logger = logging.getLogger(__name__)


class FeaturesModalityProcessor(ModalityProcessor):
    """
    Loads and preprocesses precomputed feature sequences from .npy files.

    process_sample — loads a .npy file (or passes through tensors/arrays),
                     applies optional temporal-dim permutation and frame
                     skipping; returns a [T, D] float tensor.
    process_batch  — pads a list of [T_i, D] tensors to [B, T_max, D] and
                     returns a [B, T_max] attention mask.
    """

    def __init__(
        self,
        skip_frames_stride: Optional[int] = None,
        temporal_dimension_position: int = 0,
        use_cache: bool = True,
    ):
        self.skip_frames_stride = skip_frames_stride
        self.temporal_dimension_position = temporal_dimension_position
        self.use_cache = use_cache
        if use_cache:
            self._cache_size = get_dynamic_cache_size(avg_item_size_bytes=1e6 * 0.688779)  # ~0.7 MB per feature file
            logger.info(f"Features cache size: {self._cache_size}")
            self._load_from_disk = lru_cache(maxsize=self._cache_size)(self._load_from_disk)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self, path: str) -> torch.Tensor:
        """Load a .npy feature file from disk and apply transforms."""
        features = torch.from_numpy(np.load(path))
        if self.temporal_dimension_position != 0:
            features = torch.movedim(features, self.temporal_dimension_position, 0)
        if self.skip_frames_stride is not None:
            features = frame_skipping(x=features, t_dim=0, stride=self.skip_frames_stride)
        return features

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        values — file path (str/Path), numpy array, nested list, or pre-loaded tensor.
        """
        if isinstance(values, torch.Tensor):
            return values
        if isinstance(values, np.ndarray):
            features = torch.from_numpy(values)
            if self.skip_frames_stride is not None:
                features = frame_skipping(x=features, t_dim=0, stride=self.skip_frames_stride)
            return features
        if isinstance(values, list) and all(isinstance(sub, list) for sub in values):
            features = torch.tensor(values, dtype=torch.float32)
            if self.skip_frames_stride is not None:
                features = frame_skipping(x=features, t_dim=0, stride=self.skip_frames_stride)
            return features
        if isinstance(values, (str, Path)):
            return self._load_from_disk(str(values))
        raise ValueError(f"Unsupported type for features input: {type(values)}")

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """Pad [T_i, D] tensors to [B, T_max, D] and return a [B, T_max] mask."""
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
