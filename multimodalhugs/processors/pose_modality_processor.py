from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from pose_format import Pose
    from pose_format.utils.generic import reduce_holistic, pose_hide_legs
    _POSE_FORMAT_AVAILABLE = True
except ImportError:
    _POSE_FORMAT_AVAILABLE = False

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput
from multimodalhugs.processors.utils import frame_skipping, SignalUnit


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
        signal_start_end_unit: SignalUnit = SignalUnit.MILLISECONDS,
    ):
        """
        Args:
            reduce_holistic_poses: If True, applies ``reduce_holistic`` from
                pose_format to collapse the full MediaPipe Holistic landmark set
                into a smaller, sign-language-relevant subset. Default: True.
            skip_frames_stride: If set, keeps only every N-th frame along the
                temporal axis after loading (e.g. 2 → halve frame rate).
                None disables downsampling. Default: None.
            signal_start_end_unit: Unit for ``signal_start`` / ``signal_end``
                values in the dataset.  Either ``SignalUnit.MILLISECONDS``
                (default, current behaviour — values are passed to ``Pose.read``
                as ``start_time``/``end_time``) or ``SignalUnit.FRAMES``
                (values are used as frame indices passed to ``Pose.read`` as
                ``start_frame``/``end_frame``).
                When ``signal_start=0`` and ``signal_end=0`` the full file is
                always loaded regardless of this setting.
        """
        if not _POSE_FORMAT_AVAILABLE:
            raise ImportError(
                "PoseModalityProcessor requires 'pose-format'. "
                'Install it with: pip install pose-format  or  pip install "multimodalhugs[pose]"'
            )
        try:
            signal_start_end_unit = SignalUnit(signal_start_end_unit)
        except ValueError:
            raise ValueError(
                f"Invalid signal_start_end_unit '{signal_start_end_unit}'. "
                f"Must be one of: {[u.value for u in SignalUnit]}."
            )
        self.reduce_holistic_poses = reduce_holistic_poses
        self.skip_frames_stride = skip_frames_stride
        self.signal_start_end_unit = signal_start_end_unit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pose(
        self,
        pose_file: Union[str, Path],
        signal_start: int = 0,
        signal_end: int = 0,
    ) -> torch.Tensor:
        """
        Load a .pose file and apply the full preprocessing pipeline.

        Reads the pose sequence from disk, hides leg landmarks, optionally
        reduces the holistic landmark set, normalises, and flattens landmarks
        into a feature vector per frame.

        Args:
            pose_file: Path to a binary .pose file.
            signal_start: Clip start value. When ``signal_start_end_unit`` is
                ``"milliseconds"`` this is a time in ms passed directly to
                ``Pose.read``; when ``"frames"`` it is a frame index used to
                slice the output tensor. 0 means start of file in both units.
            signal_end: Clip end value. Same unit logic as ``signal_start``.
                0 means end of file in both units.

        Returns:
            Float tensor of shape [T, D] where T is the number of frames
            (after optional downsampling) and D is the flattened landmark
            feature dimension.
        """
        if self.signal_start_end_unit == "milliseconds":
            with open(pose_file, "rb") as f:
                pose = Pose.read(
                    f,
                    start_time=signal_start or None,
                    end_time=signal_end or None,
                )
        else:
            # Pose.read natively accepts start_frame/end_frame and uses a
            # seek-capable reader, so normalization sees only the requested
            # window — consistent with the milliseconds path.
            # Note: start_frame/end_frame and start_time/end_time cannot be
            # mixed; Pose.read raises ValueError if both are set.
            start_f = int(signal_start) if signal_start else None
            end_f = int(signal_end) if signal_end else None
            with open(pose_file, "rb") as f:
                pose = Pose.read(f, start_frame=start_f, end_frame=end_f)

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
        Load and preprocess a single pose sample. Called at dataset-transform time.

        Args:
            values: One of:
                - str or Path — path to a .pose file; loaded and preprocessed.
                - torch.Tensor — returned unchanged (already preprocessed).
                - dict — mapping with keys:
                    ``"signal"`` (str/Path, required): path to the .pose file.
                    ``"signal_start"`` (int, optional): clip start in the unit
                    given by ``signal_start_end_unit``. Default 0 (start of file).
                    ``"signal_end"`` (int, optional): clip end in the unit given
                    by ``signal_start_end_unit``. Default 0 (end of file).

        Returns:
            Float tensor of shape [T, D].
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
    ) -> ProcessBatchOutput:
        """
        Pad a batch of pose tensors to a common length. Called at collation time.

        Args:
            samples: List of B tensors, each of shape [T_i, D], as returned
                by ``process_sample``.

        Returns:
            ProcessBatchOutput where:
                - data: Float tensor of shape [B, T_max, D], zero-padded.
                - mask: Bool tensor of shape [B, T_max], True for valid frames.
        """
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
