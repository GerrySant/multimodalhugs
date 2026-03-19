from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class ModalityProcessor(ABC):
    """
    Base class for single-modality processors.

    Handles one modality end-to-end: loading, preprocessing, padding, masking.
    Has no knowledge of task structure (what is encoder input vs. label, etc.).

    Two-stage interface:
      process_sample — called at dataset.with_transform() time, before batching.
                       Converts a raw value (file path, string, tensor, …) into
                       a pre-loaded tensor.  Default implementation is a no-op.
      process_batch  — called inside the DataCollator after a full batch is
                       assembled.  Pads a list of tensors to the same length and
                       returns (data_tensor, mask_tensor).  Must be implemented
                       by every concrete subclass.
    """

    def process_sample(self, values: Union[Any, Dict[str, Any]], **kwargs) -> Any:
        """
        Load and preprocess a single sample value.

        values — either:
          - a raw value (file path, string, tensor, …) for single-column slots
          - a dict {processor_param_name: value} for multi-column slots, e.g.:
              {"signal": "/path/to/file.pose", "signal_start": 0, "signal_end": 500}

        The default implementation is a no-op (returns the input unchanged),
        which is correct for modalities that need no per-sample preprocessing.
        """
        return values

    @abstractmethod
    def process_batch(
        self,
        samples: List[Any],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pad a list of pre-loaded values into a batch tensor and a mask tensor.

        Returns:
            (data_tensor, mask_tensor) where mask_tensor may be None when the
            modality produces no meaningful padding mask (e.g. labels).
        """
