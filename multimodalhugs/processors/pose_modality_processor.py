import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

try:
    from pose_format import Pose
    from pose_format.utils.generic import reduce_holistic, pose_hide_legs
    _POSE_FORMAT_AVAILABLE = True
except ImportError:
    _POSE_FORMAT_AVAILABLE = False

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput
from multimodalhugs.processors.utils import frame_skipping, SignalUnit
from multimodalhugs.processors.pose_augmentations import AUGMENTATION_REGISTRY


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
        data_augmentation_types: Optional[str] = None,
        data_augmentation_kwargs: Optional[Dict[str, Any]] = None,
        augmentation_probabilities: Optional[Dict[str, float]] = None,
        augmentation_splits: Union[str, List[str]] = "train",
        max_frames_after_augmentation: Optional[int] = None,
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
            data_augmentation_types: Comma-separated list of augmentation names
                to apply in sequence, or None / empty string for no augmentation.
                Available: ``fixed_speed_factor``, ``random_speed_perturbation``.
            data_augmentation_kwargs: Flat dict of keyword arguments forwarded to
                every selected augmentation function. Each function ignores keys
                it does not recognise. Example::

                    data_augmentation_kwargs:
                      speed_factor: 3.0          # used by fixed_speed_factor
                      min_factor: 1.0            # used by random_speed_perturbation
                      max_factor: 6.0
                      num_segments: 5

            augmentation_probabilities: Per-type application probability.
                Dict mapping augmentation name → float in [0, 1].  For each
                sample, each augmentation is applied independently with its
                own probability; types not listed default to 1.0 (always).
                Example::

                    augmentation_probabilities:
                      fixed_speed_factor: 1.0
                      random_speed_perturbation: 0.5

                Default: None (all types applied with probability 1.0).
            augmentation_splits: Dataset split(s) on which augmentation is
                applied.  Either a single split name (``"train"``) or a
                comma-separated string / list of names
                (``"train,validation"`` / ``["train", "validation"]``).
                Augmentation is skipped for splits not listed here and also
                when the sample carries no split information.
                Default: ``"train"``.
            max_frames_after_augmentation: If set, tensors longer than this
                value are randomly cropped to exactly this many frames after
                augmentation (and also after ``skip_frames_stride``).  Set
                this to the same value as ``max_frames`` in the data config
                to prevent OOM from augmentations that can increase sequence
                length (e.g. ``random_speed_perturbation`` with
                ``min_factor < 1``).  Random crop is used so that the model
                sees different parts of the sequence across epochs.
                Default: None (no cropping).
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

        # ── Augmentation setup ────────────────────────────────────────────────
        # Parse and validate augmentation type names.
        if data_augmentation_types and str(data_augmentation_types).strip():
            raw_types = [t.strip() for t in str(data_augmentation_types).split(",") if t.strip()]
            unknown = [t for t in raw_types if t not in AUGMENTATION_REGISTRY]
            if unknown:
                raise ValueError(
                    f"Unknown augmentation type(s): {unknown}. "
                    f"Available: {sorted(AUGMENTATION_REGISTRY)}."
                )
        else:
            raw_types = []

        # Public JSON-serializable attributes — included in processor_config.json.
        self.data_augmentation_types: Optional[str] = ",".join(raw_types) if raw_types else None
        self.data_augmentation_kwargs = dict(data_augmentation_kwargs or {})

        # Validate and store per-type probabilities.
        probs = dict(augmentation_probabilities or {})
        unknown_prob_keys = [k for k in probs if k not in AUGMENTATION_REGISTRY]
        if unknown_prob_keys:
            logger.warning(
                "augmentation_probabilities contains key(s) not in the registry "
                "and will be ignored: %s. Available: %s.",
                unknown_prob_keys, sorted(AUGMENTATION_REGISTRY),
            )
        invalid_probs = {k: v for k, v in probs.items() if not (0.0 <= v <= 1.0)}
        if invalid_probs:
            raise ValueError(
                f"augmentation_probabilities values must be in [0, 1]. "
                f"Got: {invalid_probs}."
            )
        self.augmentation_probabilities: Dict[str, float] = dict(augmentation_probabilities or {})

        # Normalise augmentation_splits to a list (JSON-serializable).
        if isinstance(augmentation_splits, str):
            splits = [s.strip() for s in augmentation_splits.split(",") if s.strip()]
        else:
            splits = list(augmentation_splits)
        self.augmentation_splits: List[str] = splits
        self.max_frames_after_augmentation = max_frames_after_augmentation

        # Private fast-lookup structures derived from the public attributes above.
        self._augmentation_fns_list = [AUGMENTATION_REGISTRY[t] for t in raw_types]
        self._augmentation_splits_set: frozenset = frozenset(splits)
        self._augmentation_probs_list: List[float] = [
            probs.get(name, 1.0) for name in raw_types
        ]

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

        # Some estimators (e.g. OpenPose) can detect multiple people, but in
        # sign language data there is almost always a single signer. Keeping
        # only the first person is therefore a safe default heuristic and
        # avoids downstream shape mismatches when extra detections are spurious.
        n_people = pose.body.data.shape[1]
        if n_people > 1:
            logger.warning(
                "Pose file '%s' contains %d people; truncating to person 0. "
                "This is the expected behaviour for single-signer data.",
                pose_file,
                n_people,
            )
        pose.body.data = pose.body.data[:, :1]
        pose.body.confidence = pose.body.confidence[:, :1]
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
            split = values.get("split", None)
        else:
            signal = values
            signal_start = kwargs.get("signal_start", 0)
            signal_end = kwargs.get("signal_end", 0)
            split = kwargs.get("split", None)

        if isinstance(signal, torch.Tensor):
            return signal

        tensor = self._load_pose(signal, signal_start, signal_end)

        if self._augmentation_fns_list and split in self._augmentation_splits_set:
            for fn, p in zip(self._augmentation_fns_list, self._augmentation_probs_list):
                if p >= 1.0 or torch.rand(1).item() < p:
                    tensor = fn(tensor, **self.data_augmentation_kwargs)

        if self.max_frames_after_augmentation is not None:
            T = tensor.shape[0]
            if T > self.max_frames_after_augmentation:
                start = torch.randint(0, T - self.max_frames_after_augmentation + 1, (1,)).item()
                tensor = tensor[start : start + self.max_frames_after_augmentation]

        return tensor

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
