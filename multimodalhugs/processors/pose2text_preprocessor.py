import os
import torch
import logging
import psutil

from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional, Callable, Union

from signwriting.tokenizer import normalize_signwriting
from signwriting.visualizer.visualize import signwriting_to_image

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.image_utils import PILImageResampling  # If used in 'frame_preprocessor'
from transformers.processing_utils import ProcessorMixin

from multimodalhugs.data import (
    pad_and_create_mask,
    center_image_on_white_background,
)
from multimodalhugs.processors import MultimodalSecuence2TextTranslationProcessor
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs, pose_normalization_info

logger = logging.getLogger(__name__)

def get_dynamic_cache_size():
    """Dynamically determine the LRU cache size based on available cluster memory."""
    
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
    logger.info(f"Detected a total RAM of: {total_memory_gb:.2f} GB (Source: {source})")
    
    cache_memory_fraction = 0.05  # Use 5% of available memory for cache
    avg_pose_size = 1e6 * 0.5  # Estimate ~0.5MB per pose file
    
    cache_size = int((total_memory * cache_memory_fraction) / avg_pose_size)
    return max(500, cache_size)  # Ensure a minimum cache size of 500

class Pose2TextTranslationProcessor(MultimodalSecuence2TextTranslationProcessor):  # FeatureExtractionMixin
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        reduce_holistic_poses: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        self.reduce_holistic_poses = reduce_holistic_poses
        super().__init__(tokenizer=tokenizer, **kwargs)
        
        self.use_cache = use_cache
        if self.use_cache:
            self._cache_size = get_dynamic_cache_size()
            logger.info(f"Cache size set to: {self._cache_size:,}")
            self._pose_file_to_tensor = lru_cache(maxsize=self._cache_size)(self._pose_file_to_tensor)

    def _pose_file_to_tensor(
        self, 
        pose_file: Union[str, Path], 
        source_start: int = 0, 
        source_end: int = 0
    ) -> torch.Tensor:
        """
        Converts a pose file to a tensor representation.
        
        Args:
            pose_file (Union[str, Path]): Path to the pose file.
            source_start (int): Starting time (ms) (default is 0).
            source_end (int): Ending time (ms) (default is 0).

        Returns:
            torch.Tensor: Tensor representation of the pose file.
        """
        
        with open(pose_file, "rb") as pose_file:
            pose = Pose.read(pose_file, start_time=source_start or None, end_time=source_end or None) 
        
        pose_hide_legs(pose)
    
        if self.reduce_holistic_poses:
            pose = reduce_holistic(pose)  # [t, people, d', xyz]
        
        pose = pose.normalize(pose_normalization_info(pose.header))
        tensor = pose.torch().body.data.zero_filled()
        return tensor.contiguous().view(tensor.size(0), -1)

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_sequences = [self._pose_file_to_tensor(sample["source"], sample["source_start"], sample["source_end"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_sequences)
        return {
            "inputs_embeds": padded_inputs,
            "attention_mask": padded_input_masks
        }, kwargs
