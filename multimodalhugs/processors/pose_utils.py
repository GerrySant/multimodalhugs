"""
Utility functions for pose sequence analysis.

All functions assume the pose has been processed by ``reduce_holistic``, giving
the fixed component layout: POSE(8) + FACE(128) + LEFT_HAND(21) + RIGHT_HAND(21)
= 178 keypoints.
"""

from typing import List, Tuple

import numpy as np
import numpy.ma as ma

# Re-use the layout constants from pose_augmentations
from multimodalhugs.processors.pose_augmentations import (
    _POSE_LELBOW_IDX,
    _POSE_RELBOW_IDX,
    _POSE_LWRIST_IDX,
    _POSE_RWRIST_IDX,
    _HAND_START,
    _LHAND_N,
)

# Proximal finger joints used for motion energy (index + middle MCP on each hand)
_MOTION_KPTS = [
    _POSE_LELBOW_IDX,
    _POSE_RELBOW_IDX,
    _POSE_LWRIST_IDX,
    _POSE_RWRIST_IDX,
    _HAND_START + 5,           # left hand index MCP
    _HAND_START + 9,           # left hand middle MCP
    _HAND_START + _LHAND_N + 5,  # right hand index MCP
    _HAND_START + _LHAND_N + 9,  # right hand middle MCP
]

_Y = 1  # coordinate axis index for vertical position


def _savgol(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    from scipy.signal import savgol_filter
    window = int(fps * seconds) | 1  # must be odd
    return np.clip(savgol_filter(np.nan_to_num(x), window, polyorder=2), 0, None)


def _is_rest_frame(data: np.ndarray) -> bool:
    """
    Return True if the frame looks like a rest posture (both wrists below their
    respective elbows in image coordinates, where y increases downward).
    Returns False if any of the four keypoints is NaN / undetected.
    """
    l_elbow_y = data[_POSE_LELBOW_IDX, _Y]
    l_wrist_y = data[_POSE_LWRIST_IDX, _Y]
    r_elbow_y = data[_POSE_RELBOW_IDX, _Y]
    r_wrist_y = data[_POSE_RWRIST_IDX, _Y]
    if any(np.isnan(v) for v in [l_elbow_y, l_wrist_y, r_elbow_y, r_wrist_y]):
        return False
    return (l_wrist_y > l_elbow_y) and (r_wrist_y > r_elbow_y)


def segment_pose_into_signs(
    pose,
    min_sign_duration: float = 0.8,
    drop_final_rest: bool = True,
) -> List[Tuple[int, int]]:
    """
    Segment a pose sequence into individual signs by analysing wrist/arm velocity.

    Algorithm
    ---------
    1. Compute per-frame velocity from 8 upper-body keypoints (both elbows,
       both wrists, proximal finger joints on each hand), taking the median
       across keypoints so that missing detections don't dominate.
    2. Apply three successive Savitzky-Golay smoothing passes (0.15 s → 0.4 s →
       0.8 s) to merge sub-sign micro-movements and internal repetitions into
       a single activity "blob" per sign.
    3. Find peaks in the smoothed signal; each peak corresponds to one sign.
       Peaks must be at least ``min_sign_duration`` seconds apart and above the
       mean velocity level.
    4. Set sign boundaries at the velocity valley between consecutive peaks.
    5. Optionally drop the last segment if it is a rest transition: if the last
       frame of the clip is in a rest posture (both wrists below elbows), the
       final detected segment is the signer returning to rest, not a real sign.

    Args:
        pose:               A ``pose_format.Pose`` object after ``reduce_holistic``
                            has been applied (178 keypoints).
        min_sign_duration:  Minimum time in seconds between two sign peaks.
                            Controls the temporal resolution — signs faster than
                            this cannot be distinguished. Default: 0.8 s.
        drop_final_rest:    If True (default), discard the last segment when the
                            final frame of the clip is in a rest posture, treating
                            it as the signer's return to neutral rather than a sign.

    Returns:
        List of ``(start_frame, end_frame)`` tuples (both inclusive) in the
        coordinate frame of the input pose (frame 0 = first frame of ``pose``).
        Returns an empty list if no movement above the mean level is detected.
    """
    from scipy.signal import find_peaks

    fps  = float(pose.body.fps)
    data = ma.filled(pose.body.data[:, 0, :, :], np.nan)  # [T, K, C]
    T    = data.shape[0]

    if T < 2:
        return [(0, T - 1)] if T == 1 else []

    # ── 1. Multi-keypoint composite velocity ──────────────────────────────────
    kpt_data = data[:, _MOTION_KPTS, :2]  # [T, N, 2]
    kpt_vel  = np.linalg.norm(np.diff(kpt_data, axis=0), axis=2)  # [T-1, N]
    vel      = np.nanmedian(kpt_vel, axis=1)  # [T-1]

    # ── 2. Three-stage smoothing ───────────────────────────────────────────────
    vel_coarse = _savgol(_savgol(_savgol(vel, fps, 0.15), fps, 0.4), fps, 0.8)

    # ── 3. Peak detection ─────────────────────────────────────────────────────
    min_dist = max(1, int(fps * min_sign_duration))
    peaks, _ = find_peaks(vel_coarse, distance=min_dist, height=vel_coarse.mean())

    if len(peaks) == 0:
        return []

    # ── 4. Boundaries at valleys between consecutive peaks ────────────────────
    boundaries = [0]
    for i in range(len(peaks) - 1):
        segment = vel_coarse[peaks[i] : peaks[i + 1]]
        valley  = int(peaks[i]) + int(np.argmin(segment))
        boundaries.append(valley)
    boundaries.append(T - 1)

    segments = list(zip(boundaries[:-1], boundaries[1:]))

    # ── 5. Drop final rest transition ─────────────────────────────────────────
    if drop_final_rest and len(segments) > 1:
        if _is_rest_frame(data[-1]):
            segments = segments[:-1]

    return segments
