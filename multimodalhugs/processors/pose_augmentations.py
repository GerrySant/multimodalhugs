"""
Augmentations for pose frame sequences.

Each function takes a [T, D] float tensor and returns a [T', D] tensor.
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


# Component layout after reduce_holistic (always in this order):
_POSE_N  =   8   # POSE_LANDMARKS: LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW,
               #                  RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP
_FACE_N  = 128   # FACE_LANDMARKS
_LHAND_N =  21   # LEFT_HAND_LANDMARKS  (index 0 = WRIST)
_RHAND_N =  21   # RIGHT_HAND_LANDMARKS (index 0 = WRIST)
_N_COORDS =  3   # x, y, z per keypoint

_HAND_START = _POSE_N + _FACE_N                   # 136
_HAND_END   = _HAND_START + _LHAND_N + _RHAND_N  # 178

# POSE_LANDMARKS keypoint indices (verified from reduce_holistic output):
_POSE_LSHOULDER_IDX = 0
_POSE_RSHOULDER_IDX = 1
_POSE_LELBOW_IDX    = 2
_POSE_RELBOW_IDX    = 3
_POSE_LWRIST_IDX    = 4  # must stay in sync with LEFT_HAND[0]
_POSE_RWRIST_IDX    = 5  # must stay in sync with RIGHT_HAND[0]


def apply_component_jitter(
    x: torch.Tensor,
    global_jitter_std: float = 0.02,
    hand_jitter_std: float = 0.01,
    **kwargs,
) -> torch.Tensor:
    """
    Add a per-sequence spatial jitter at the component level.

    Samples one (Δx, Δy, Δz) offset vector per component and adds it uniformly
    to all keypoints in that component, keeping the same offset for every frame
    in the sequence. This preserves handshape (relative finger positions) and
    simulates natural variation in signer position across different recordings
    of the same sign.

    Two levels of jitter are applied independently:

    - Global jitter (``global_jitter_std``): a single offset added to every
      keypoint in the pose. Recommended value: 0 (disabled). Because
      ``pose.normalize()`` already centers every sequence on the shoulder
      midpoint, all poses in both training and inference are anchored at
      (0, 0) by construction. A global shift therefore introduces variation
      that does not exist in the real data distribution and creates a
      training/inference mismatch. The natural per-frame drift of the shoulder
      midpoint after normalization is only ~0.007–0.016 shoulder-widths std,
      so any meaningful ``global_jitter_std`` value exceeds the real variation.
    - Hand-local jitter (``hand_jitter_std``): an additional, independent offset
      added only to the left-hand and right-hand components (each hand gets its
      own independent offset). The POSE wrist keypoints (LEFT_WRIST index 4,
      RIGHT_WRIST index 5) receive the same offset so both representations of
      the wrist stay consistent. The elbow is left at its original position;
      the forearm length therefore changes by the jitter magnitude, which is
      acceptable because pose estimation from video cannot reliably recover the
      depth dimension and real sequences already carry natural arm-length
      variation across frames.

    Coordinate scale and normalization
    -----------------------------------
    This augmentation is designed to run after ``pose.normalize()``, which is
    always applied in ``_load_pose`` before the tensor reaches any augmentation
    function. After normalization, all coordinates are expressed in
    shoulder-width units (the mean shoulder-to-shoulder distance equals 1.0),
    centered on the torso midpoint. Empirically, the full pose spans roughly
    [-1.3, 1.3] in x and [-1.2, 1.8] in y across our datasets.

    This means std values should be interpreted in shoulder-width units:
      - global_jitter_std=0.02 shifts the whole pose by ~2% of a shoulder-width
      - global_jitter_std=0.05 shifts by ~5% of a shoulder-width
      - hand_jitter_std=0.01  adds ~1% of a shoulder-width per hand

    Note: applying jitter after normalization intentionally breaks the exact
    centering invariant of ``pose.normalize()`` — the pose is no longer centered
    on the shoulder midpoint. This is desirable: it is precisely the source of
    the positional variation the augmentation is meant to introduce.

    Assumes the pose has been processed by ``reduce_holistic``, giving the fixed
    component layout: POSE(8) + FACE(128) + LEFT_HAND(21) + RIGHT_HAND(21) = 178
    keypoints, D = 534.

    Args:
        x:                 [T, D] pose tensor (D = 534 after reduce_holistic).
        global_jitter_std: Std of the global offset in shoulder-width units.
                           Recommended: 0 (see note above). Default: 0.02.
        hand_jitter_std:   Std of the per-hand additional offset in
                           shoulder-width units. Set to 0 to disable.
                           Default: 0.01.
    """
    if global_jitter_std < 0 or hand_jitter_std < 0:
        raise ValueError("global_jitter_std and hand_jitter_std must be >= 0")

    T, D = x.shape
    n_kpts = D // _N_COORDS

    out = x.clone().float().view(T, n_kpts, _N_COORDS)

    if global_jitter_std > 0:
        # Shape [1, 1, 3] broadcasts over [T, K, 3]
        out = out + torch.randn(1, 1, _N_COORDS) * global_jitter_std

    if hand_jitter_std > 0:
        left_offset  = torch.randn(_N_COORDS) * hand_jitter_std  # [3]
        right_offset = torch.randn(_N_COORDS) * hand_jitter_std  # [3]

        out[:, _HAND_START : _HAND_START + _LHAND_N, :] += left_offset
        out[:, _HAND_START + _LHAND_N : _HAND_END,  :] += right_offset
        out[:, _POSE_LWRIST_IDX, :] += left_offset
        out[:, _POSE_RWRIST_IDX, :] += right_offset

    return out.view(T, D).to(x.dtype)


def apply_gaussian_noise(
    x: torch.Tensor,
    noise_std: float = 0.01,
    noise_mean: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Add independent Gaussian noise to every keypoint coordinate at every frame.

    Prevents the model from seeing identical pose sequences for the same sign
    across epochs. Coordinates after normalization are in shoulder-width units
    (roughly [-1.5, 1.5]), so noise_std should be chosen relative to that scale:
    0.01 is a subtle perturbation, 0.05 is more noticeable.

    Args:
        x:          [T, D] pose tensor.
        noise_std:  Standard deviation of the Gaussian noise. Must be >= 0.
        noise_mean: Mean of the noise distribution. Default: 0.0.
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")
    if noise_std == 0.0:
        return x
    return x + torch.randn_like(x) * noise_std + noise_mean


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    "fixed_speed_factor":        apply_fixed_speed_factor,
    "random_speed_perturbation": apply_random_speed_perturbation,
    "component_jitter":          apply_component_jitter,
    "gaussian_noise":            apply_gaussian_noise,
}
