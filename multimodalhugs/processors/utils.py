import os

import psutil
import torch


def get_dynamic_cache_size(avg_item_size_bytes: float) -> int:
    """
    Estimate a sensible LRU cache size from available memory.

    Uses SLURM job-allocation memory variables when running on a cluster,
    falling back to the machine's total virtual memory. Targets 5% of
    available memory divided by the average size of one cached item.

    Args:
        avg_item_size_bytes: Estimated average size of one cached item in bytes.

    Returns:
        Maximum number of items to cache (at least 10).
    """
    cluster_mem = None
    for var in ("SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        if os.getenv(var):
            cluster_mem = int(os.getenv(var)) * 1e6
            break
    total = cluster_mem or psutil.virtual_memory().total
    return max(10, int((total * 0.05) / avg_item_size_bytes))


def frame_skipping(x: torch.Tensor, t_dim: int, stride: int) -> torch.Tensor:
    """
    Downsample the temporal dimension of a tensor by a fixed stride.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        t_dim (int): Index of the temporal dimension.
        stride (int): Sampling stride (keep every `stride`-th element).

    Returns:
        torch.Tensor: Tensor with the same shape except the temporal dimension
                      is reduced to ceil(orig_length/stride).
    """
    # build slice objects for all dims, stepping the temporal dim by `stride`
    slices = [slice(None)] * x.dim()
    slices[t_dim] = slice(None, None, stride)
    return x[tuple(slices)]