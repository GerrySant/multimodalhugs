"""
Temporal speed augmentations for pose frame sequences.

Each function takes a [T, D] float tensor and returns a resampled [T', D] tensor.
All functions accept **kwargs so they can be called uniformly from a registry with
a shared flat kwargs dict — unknown keys are silently ignored.
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Core utility
# ──────────────────────────────────────────────────────────────────────────────

def resample_temporal(x: torch.Tensor, T_new: int) -> torch.Tensor:
    """
    Linearly resample a [T, D] tensor to [T_new, D] along the time axis.

    Uses F.interpolate (1-D linear) which handles both upsampling and
    downsampling and preserves all feature dimensions without looping.
    """
    if T_new == x.shape[0]:
        return x
    # F.interpolate expects [N, C, L]; we treat D as channels, T as length.
    out = F.interpolate(
        x.T.unsqueeze(0).float(),  # [1, D, T]
        size=T_new,
        mode="linear",
        align_corners=False,
    )
    return out.squeeze(0).T  # [T_new, D]


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation functions
# ──────────────────────────────────────────────────────────────────────────────

def apply_fixed_speed_factor(
    x: torch.Tensor,
    speed_factor: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Speed up (or slow down) the sequence by a fixed multiplicative factor.

    The output has round(T / speed_factor) frames. A factor > 1 produces
    fewer frames (faster signing); < 1 produces more frames (slower).

    Args:
        x:            [T, D] pose tensor.
        speed_factor: Divisor applied to the frame count. Must be > 0.
    """
    if speed_factor <= 0:
        raise ValueError(f"speed_factor must be > 0, got {speed_factor}")
    T = x.shape[0]
    T_new = max(1, round(T / speed_factor))
    return resample_temporal(x, T_new)


def apply_random_speed_perturbation(
    x: torch.Tensor,
    min_factor: float = 1.0,
    max_factor: float = 6.0,
    num_segments: int = 5,
    min_num_segments: Optional[int] = None,
    max_num_segments: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply locally-varying speed changes inspired by DTW-based augmentation.

    Divides the sequence into randomly placed segments, each independently
    resampled with a speed factor sampled uniformly from [min_factor, max_factor].

    The number of segments is either fixed (``num_segments``) or itself drawn
    uniformly from [``min_num_segments``, ``max_num_segments``] each call,
    adding an extra layer of structural variability.

    Args:
        x:                [T, D] pose tensor.
        min_factor:       Lower bound for the per-segment speed factor.
        max_factor:       Upper bound for the per-segment speed factor.
        num_segments:     Fixed number of segments. Used only when neither
                          ``min_num_segments`` nor ``max_num_segments`` is set.
                          Default: 5.
        min_num_segments: Lower bound for the randomly sampled segment count
                          (inclusive). Requires ``max_num_segments``.
        max_num_segments: Upper bound for the randomly sampled segment count
                          (inclusive). When set, ``num_segments`` is ignored
                          and the count is drawn from
                          [``min_num_segments``, ``max_num_segments``].
    """
    if min_factor <= 0 or max_factor <= 0:
        raise ValueError("min_factor and max_factor must be > 0")
    if min_factor > max_factor:
        raise ValueError("min_factor must be <= max_factor")

    T, D = x.shape

    # Resolve number of segments for this call.
    if max_num_segments is not None:
        lo = min_num_segments if min_num_segments is not None else 1
        hi = max_num_segments
        if lo > hi:
            raise ValueError("min_num_segments must be <= max_num_segments")
        num_segments = int(torch.randint(lo, hi + 1, (1,)))

    if T < 2 or num_segments < 1:
        return x

    n_segs = min(num_segments, T)

    # O(n_segs) boundary sampling — avoids the O(T) randperm used in naive
    # implementations. torch.unique handles the rare case of duplicate draws.
    if n_segs > 1:
        cuts = torch.unique(torch.randint(1, T, (n_segs - 1,)).sort()[0])
        boundaries = torch.cat([torch.zeros(1, dtype=torch.long), cuts, torch.tensor([T])])
        n_segs = len(boundaries) - 1
    else:
        boundaries = torch.tensor([0, T])

    # Sample all per-segment factors in one call.
    factors = (min_factor + (max_factor - min_factor) * torch.rand(n_segs)).tolist()

    segments = []
    for i in range(n_segs):
        start = int(boundaries[i])
        end   = int(boundaries[i + 1])
        if start >= end:
            continue
        seg = x[start:end]
        seg_len_new = max(1, round((end - start) / factors[i]))
        segments.append(resample_temporal(seg, seg_len_new))

    return torch.cat(segments, dim=0) if segments else x


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    "fixed_speed_factor":        apply_fixed_speed_factor,
    "random_speed_perturbation": apply_random_speed_perturbation,
}
