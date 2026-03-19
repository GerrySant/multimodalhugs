from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor
from multimodalhugs.processors.utils import frame_skipping


class PoseModalityProcessor(ModalityProcessor):
    """
    Loads and preprocesses pose sequences from .pose files.

    process_sample — reads a .pose file, applies hide_legs / reduce_holistic /
                     normalization, returns a [T, D] float tensor.
    process_batch  — pads a list of [T_i, D] tensors to [B, T_max, D] and
                     returns a [B, T_max] attention mask.
    """

    def __init__(
        self,
        reduce_holistic_poses: bool = True,
        skip_frames_stride: Optional[int] = None,
    ):
        self.reduce_holistic_poses = reduce_holistic_poses
        self.skip_frames_stride = skip_frames_stride

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pose(
        self,
        pose_file: Union[str, Path],
        signal_start: int = 0,
        signal_end: int = 0,
    ) -> torch.Tensor:
        with open(pose_file, "rb") as f:
            pose = Pose.read(
                f,
                start_time=signal_start or None,
                end_time=signal_end or None,
            )
        pose_hide_legs(pose)
        if self.reduce_holistic_poses:
            pose = reduce_holistic(pose)
        pose = pose.normalize()
        tensor = pose.torch().body.data.zero_filled()
        tensor = tensor.contiguous().view(tensor.size(0), -1)
        if self.skip_frames_stride is not None:
            tensor = frame_skipping(x=tensor, t_dim=0, stride=self.skip_frames_stride)
        return tensor

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        values — file path (str/Path) or pre-loaded tensor, or a dict:
                   {"signal": path, "signal_start": int, "signal_end": int}
        """
        if isinstance(values, dict):
            signal = values["signal"]
            signal_start = values.get("signal_start", 0)
            signal_end = values.get("signal_end", 0)
        else:
            signal = values
            signal_start = kwargs.get("signal_start", 0)
            signal_end = kwargs.get("signal_end", 0)

        if isinstance(signal, torch.Tensor):
            return signal

        return self._load_pose(signal, signal_start, signal_end)

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad [T_i, D] tensors to [B, T_max, D] and return a [B, T_max] mask.
        """
        padded, mask = pad_and_create_mask(samples)
        return padded, mask
