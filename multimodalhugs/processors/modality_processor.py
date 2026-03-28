from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch


class ProcessBatchOutput(NamedTuple):
    """
    Typed return value from ModalityProcessor.process_batch.

    data — batched tensor of shape [B, ...].
    mask — attention / padding mask (shape [B, T] or similar), or None.
           None signals "no mask to add", not "all positions are valid".
           It is the correct return value when the modality produces no
           variable-length padding (e.g. label sequences padded with -100,
           fixed-size embeddings).  Callers must check for None before
           inserting the mask into a model batch.
    """
    data: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]


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
                       returns a ProcessBatchOutput(data, mask).  Must be
                       implemented by every concrete subclass.

    Contract for process_sample return values:
      Subclasses may return any value that their process_batch implementation
      accepts in the input list — typically a torch.Tensor after per-sample
      loading, or the raw value unchanged for modalities that batch in one step
      (e.g. text).  No runtime type check is performed at the base-class level;
      type mismatches surface inside process_batch with a clear error from the
      underlying tensor operation.
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
    ) -> ProcessBatchOutput:
        """
        Pad a list of pre-loaded values into a batch tensor and a mask tensor.

        Returns:
            ProcessBatchOutput(data, mask) where:
              data — batched tensor, shape [B, ...].
              mask — attention/padding mask, or None.
                     Return None when the modality produces no variable-length
                     padding (e.g. labels padded with -100, or fixed-size
                     embeddings).  None means "no mask to add to the batch",
                     not "all positions are valid" — callers must not insert a
                     None mask into a tensor batch without a None check.
        """

    def __repr__(self) -> str:
        """Show class name and JSON-serializable constructor attributes."""
        _SKIP = {"tokenizer", "pretrained_tokenizer", "custom_preprocessor"}
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v) and k not in _SKIP
        )
        return f"{self.__class__.__name__}({params})"
