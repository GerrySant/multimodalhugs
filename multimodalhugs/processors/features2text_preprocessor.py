import os
import torch
import psutil
import logging
import numpy as np


from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional, Callable, Union
from transformers.processing_utils import ProcessorMixin

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor
from multimodalhugs.processors.utils import frame_skipping

logger = logging.getLogger(__name__)

def get_dynamic_cache_size():
    """Dynamically determine the LRU cache size based on available cluster memory."""
    
    # Check for memory allocation in different job schedulers
    cluster_mem = None
    source = "System Memory"  # Default fallback
    
    if os.getenv("SLURM_MEM_PER_GPU") or os.getenv("SLURM_MEM_PER_NODE") or os.getenv("SLURM_MEM_PER_CPU"):
        cluster_mem = os.getenv("SLURM_MEM_PER_GPU") or os.getenv("SLURM_MEM_PER_NODE") or os.getenv("SLURM_MEM_PER_CPU")
        source = "SLURM"
    elif os.getenv("PBS_NODEFILE"):
        cluster_mem = os.getenv("PBS_MEMORY")  # PBS memory is usually set per node
        source = "PBS"
    elif os.getenv("SGE_HGR_memory_requested"):
        cluster_mem = os.getenv("SGE_HGR_memory_requested")  # Memory per host group
        source = "SGE"
    elif os.getenv("LSB_MJOBID"):
        cluster_mem = os.getenv("LSB_DJOB_MEMLIMIT")  # LSF memory limit per job
        source = "LSF"
    
    if cluster_mem:
        total_memory = int(cluster_mem) * 1e6  # Convert MB to bytes
    else:
        total_memory = psutil.virtual_memory().total  # Fallback to system RAM
    
    total_memory_gb = total_memory / 1e9  # Convert bytes to GB
    logger.info(f"Detected a total RAM of: {total_memory_gb:.2f} GB (Source: {source})")  # Display in GB
    
    cache_memory_fraction = 0.05  # Use 5% of available memory for cache
    avg_feature_size = 1e6 * 0.688779 # Estimate ~1MB per feature file

    cache_size = int((total_memory * cache_memory_fraction) / avg_feature_size)
    return max(500, cache_size)  # Ensure a minimum cache size of 500

class Features2TextTranslationProcessor(MultimodalSecuence2TextTranslationProcessor):
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        use_cache: bool = True,
        skip_frames_stride: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.use_cache = use_cache
        self.skip_frames_stride = skip_frames_stride
        if self.use_cache:
            # Dynamically determine cache size
            self._cache_size = get_dynamic_cache_size()
            logger.info(f"Cache size set to: {self._cache_size:,}")  # Format with thousands separator

            # Overwrite _features_file_to_tensor with a new cached version
            self._features_file_to_tensor = lru_cache(maxsize=self._cache_size)(self._features_file_to_tensor)

    def _features_file_to_tensor(self, features_file: Union[str, Path, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Loads feature tensor from .npy file, or returns it directly if it is already a tensor."""
        if isinstance(features_file, torch.Tensor):
            # Case in which the transformation has already been performed during the dataset._get_items_()
            return features_file
        elif isinstance(features_file, np.ndarray):
            features = torch.from_numpy(features_file)
        elif isinstance(features_file, (str, Path)):
            features = torch.from_numpy(np.load(features_file))
        elif isinstance(features_file, list) and all(isinstance(sublist, list) for sublist in features_file):
            features = torch.tensor(features_file, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported type for features_file: {type(features_file)}")
        features = frame_skipping(x=features, t_dim=0, stride=self.skip_frames_stride) if self.skip_frames_stride is not None else features
        return features


    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_sequences = [self._features_file_to_tensor(sample["signal"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_sequences)
        return {
            "input_frames": padded_inputs,
            "attention_mask": padded_input_masks
        }, kwargs

    def _transform_get_items_output(self, batch):
        """
        Returns a transformation function applied at the dataset level during iteration.

        This method defines a transformation that is applied to each batch **within the dataset iterator**, 
        typically by using `datasets.Dataset.with_transform()`. As a result, the transformation is executed 
        at runtime during `__getitem__()` or `__getitems__()`, which allows it to benefit from prefetching 
        and parallel data loading when using multiple DataLoader workers.

        Unlike the `_obtain_*` methods, which are also executed on-the-fly but within the **processor call 
        (typically inside the DataCollator)**, this transformation occurs **prior to batching and collation**. 
        It is therefore ideal for operations that are expensive and can be parallelized at the sample or batch 
        level, such as decoding signals, loading external files, or converting inputs to intermediate formats.

        Use this method to preprocess inputs early in the pipeline while maintaining a modular design that 
        separates dataset-level and collator-level responsibilities.

        Args:
            batch (Dict[str, List[Any]]): A dictionary representing a batch of dataset examples (not yet collated).

        Returns:
            Dict[str, List[Any]]: The transformed batch, with updated or added fields ready for collation.
        """
        tensor_signals = [self._features_file_to_tensor(batch["signal"][i]) for i in range(len(batch["signal"]))]
        batch["signal"] = tensor_signals
        return batch