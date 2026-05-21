"""
Visualise the augmentation defined in a training config YAML.

Reads the PoseModalityProcessor slot from the config, instantiates the
processor with the exact same kwargs used during training, and applies
the augmentation to a real .pose sample.  Saves:

  <stem>_original.pose            — preprocessed clip (no augmentation)
  <stem>_augmented_seed<n>.pose   — one augmented sample per seed
  <stem>_augmentation_comparison.png

Coordinate space of the saved .pose files
------------------------------------------
All saved .pose files are in the original pixel coordinate space and open
correctly in any viewer, regardless of augmentation type.

For spatial augmentations (component_jitter, gaussian_noise) the script
internally works in normalized coordinates (shoulder-width units), but applies
the inverse normalization transform before writing the file so the output
coordinates match the original pixel space. The augmentation effect is
therefore visible at the correct scale in any pose viewer.

Usage
-----
    conda activate bt-slt-aug

    python debug_run_scripts/debug/visualize_augmentation.py \\
        --config debug_run_scripts/debug/pretraining.yaml \\
        --pose_file /shares/.../poses/18/chunk_000285.pose \\
        --signal_start 3820 --signal_end 4676

    python debug_run_scripts/debug/visualize_augmentation.py \\
        --config /path/to/pretraining_augmented.yaml \\
        --pose_file /path/to/sample.pose \\
        --n_samples 5 \\
        --out_dir /tmp/aug_debug
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.ma as ma
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ── Repo on path ──────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_POSE_FILE    = (
    "/shares/sigma.ebling.cl.uzh/common/SLT_BT_project/"
    "BT_data/srf_BT_22/poses/18/chunk_000285.pose"
)
DEFAULT_SIGNAL_START = 3820
DEFAULT_SIGNAL_END   = 4676
DEFAULT_OUT_DIR      = "/tmp/aug_debug"

# Augmentation types that require normalized coordinates to produce meaningful
# results. When any of these appear in the config, the script applies all
# augmentations to the normalized tensor.
SPATIAL_AUG_TYPES = {"component_jitter", "gaussian_noise", "appearance_transfer"}


# ── Config parsing ────────────────────────────────────────────────────────────

def extract_pose_processor_kwargs(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    slots = cfg.get("processor", {}).get("slots", [])
    for slot in slots:
        if slot.get("processor_class") == "PoseModalityProcessor":
            return dict(slot.get("processor_kwargs", {}))
    raise ValueError(
        f"No PoseModalityProcessor slot found in {config_path}. "
        f"Available classes: {[s.get('processor_class') for s in slots]}"
    )


def needs_normalization(proc) -> bool:
    """Return True if any configured augmentation requires normalized coordinates."""
    aug_types = set(proc.data_augmentation_types.split(",")) if proc.data_augmentation_types else set()
    return bool(aug_types & SPATIAL_AUG_TYPES)


def _inv_norm_params(raw_pose, norm_pose) -> Tuple[float, float, float]:
    """
    Recover the linear transform raw ≈ norm * scale + (cx, cy) via OLS.

    pose.normalize() applies: norm = (raw - center) / scale
    Inverse:                  raw  = norm * scale + center

    Used to convert augmented normalized coordinates back to pixel space
    before writing .pose files so they render at the correct scale in viewers.
    """
    raw_x = np.ma.filled(raw_pose.body.data[:, 0, :, 0], np.nan).ravel()
    nor_x = np.ma.filled(norm_pose.body.data[:, 0, :, 0], np.nan).ravel()
    raw_y = np.ma.filled(raw_pose.body.data[:, 0, :, 1], np.nan).ravel()
    nor_y = np.ma.filled(norm_pose.body.data[:, 0, :, 1], np.nan).ravel()

    ok = (raw_x != 0) & (nor_x != 0) & np.isfinite(raw_x) & np.isfinite(nor_x)
    if ok.sum() < 4:
        return 1.0, 0.0, 0.0

    A = np.stack([nor_x[ok], np.ones(ok.sum())], axis=1)
    (scale, cx), *_ = np.linalg.lstsq(A, raw_x[ok], rcond=None)

    ok_y = np.isfinite(raw_y) & np.isfinite(nor_y)
    cy = float(np.nanmean(raw_y[ok_y] - scale * nor_y[ok_y])) if ok_y.any() else cx

    return float(scale), float(cx), float(cy)


# ── Pose loading ──────────────────────────────────────────────────────────────

def load_pose(
    pose_file: str,
    signal_start: int,
    signal_end: int,
    signal_start_end_unit: str = "frames",
    reduce_holistic_poses: bool = True,
    normalize: bool = False,
) -> Tuple[Any, torch.Tensor, Tuple[int, int, int], Optional[Tuple[float, float, float]]]:
    """
    Load a .pose file and apply structural preprocessing.

    When normalize=False (default for temporal-only augmentations): hide_legs +
    reduce_holistic only.

    When normalize=True (required for spatial augmentations): additionally
    applies pose.normalize() and records the inverse transform parameters
    (scale, cx, cy) so saved .pose files can be denormalized back to pixel
    space before writing, keeping them renderable in any viewer.

    Returns
    -------
    pose       : Pose object after preprocessing (normalized if normalize=True)
    flat       : [T, D] float32 tensor ready for augmentation
    body_shape : (people, keypoints, coords) for reshaping back
    inv_norm   : (scale, cx, cy) to undo normalization, or None
    """
    from pose_format import Pose
    from pose_format.utils.generic import reduce_holistic, pose_hide_legs

    unit = str(signal_start_end_unit).lower()
    with open(pose_file, "rb") as f:
        if unit == "frames":
            pose = Pose.read(f, start_frame=signal_start or None, end_frame=signal_end or None)
        else:
            pose = Pose.read(f, start_time=signal_start or None, end_time=signal_end or None)

    pose.body.data = pose.body.data[:, :1]
    pose.body.confidence = pose.body.confidence[:, :1]
    pose_hide_legs(pose)
    if reduce_holistic_poses:
        pose = reduce_holistic(pose)

    inv_norm = None
    if normalize:
        pre_norm = copy.deepcopy(pose)
        pose = pose.normalize()
        inv_norm = _inv_norm_params(pre_norm, pose)

    body_tensor = pose.torch().body.data.zero_filled()  # [T, people, K, C]
    _, people, keypoints, coords = body_tensor.shape
    flat = body_tensor.contiguous().view(body_tensor.size(0), -1).float()
    return pose, flat, (people, keypoints, coords), inv_norm


def augment(flat: torch.Tensor, proc) -> torch.Tensor:
    """Apply all augmentation functions unconditionally (split check bypassed)."""
    result = flat.clone()
    for fn in proc._augmentation_fns_list:
        result = fn(result, **proc.data_augmentation_kwargs)
    return result


def tensor_to_pose(
    original_pose,
    augmented_flat: torch.Tensor,
    body_shape: Tuple[int, int, int],
    inv_norm: Optional[Tuple[float, float, float]] = None,
) -> Any:
    """
    Reconstruct a Pose object from an augmented [T', D] tensor.

    If inv_norm=(scale, cx, cy) is provided, applies the inverse of
    pose.normalize() before returning, so the saved file is in pixel space.
    """
    people, keypoints, coords = body_shape
    T_new = augmented_flat.shape[0]
    T_old = original_pose.body.data.shape[0]

    orig_dtype = original_pose.body.data.dtype
    body_np = (
        augmented_flat.view(T_new, people, keypoints, coords)
        .numpy().astype(float)
    )

    if inv_norm is not None:
        scale, cx, cy = inv_norm
        body_np[..., 0] = body_np[..., 0] * scale + cx
        body_np[..., 1] = body_np[..., 1] * scale + cy
        if coords > 2:
            body_np[..., 2] *= scale

    conf_old = torch.from_numpy(original_pose.body.confidence.copy()).float()
    conf_rs = F.interpolate(
        conf_old.permute(1, 2, 0).reshape(1, people * keypoints, T_old).float(),
        size=T_new, mode="linear", align_corners=False,
    ).squeeze(0).reshape(people, keypoints, T_new).permute(2, 0, 1).numpy()

    new_pose = copy.deepcopy(original_pose)
    new_pose.body.data = ma.MaskedArray(body_np.astype(orig_dtype), mask=False)
    new_pose.body.confidence = conf_rs.astype(np.float32)
    return new_pose


def save_pose(pose, path: Path):
    with open(path, "wb") as f:
        pose.write(f)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(ax, flat: torch.Tensor, title: str, color: str,
                    x_max: int, coord_label: str):
    per_frame = flat.abs().mean(dim=1).numpy()
    ax.plot(per_frame, color=color, linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_ylabel(f"Mean |keypoint| ({coord_label})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, x_max)
    ax.text(0.97, 0.95, f"T={len(per_frame)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=color)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise pose augmentation from a training config YAML."
    )
    parser.add_argument("--config", required=True,
                        help="Path to a training YAML with a PoseModalityProcessor slot.")
    parser.add_argument("--pose_file",    default=DEFAULT_POSE_FILE)
    parser.add_argument("--signal_start", type=int, default=DEFAULT_SIGNAL_START)
    parser.add_argument("--signal_end",   type=int, default=DEFAULT_SIGNAL_END)
    parser.add_argument("--out_dir",      default=DEFAULT_OUT_DIR)
    parser.add_argument("--n_samples",    type=int, default=3,
                        help="Number of augmented samples (useful for stochastic augmentations).")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Parse config and build processor ─────────────────────────────────────
    print(f"Config      : {args.config}")
    proc_kwargs = extract_pose_processor_kwargs(args.config)
    print("Processor kwargs extracted:")
    for k, v in proc_kwargs.items():
        print(f"  {k}: {v}")

    from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor
    proc = PoseModalityProcessor(**proc_kwargs)

    if not proc._augmentation_fns_list:
        print("\nNo augmentation types configured in this config — nothing to visualise.")
        return

    aug_names = proc.data_augmentation_types
    normalize = needs_normalization(proc)
    coord_label = "shoulder-width units" if normalize else "pixels"

    print(f"\nAugmentation : {aug_names}")
    print(f"Kwargs       : {proc.data_augmentation_kwargs}")
    print(f"Probs        : {proc.augmentation_probabilities}")
    print(f"Splits       : {proc.augmentation_splits}")
    print(f"Normalize    : {normalize}  → coordinates in {coord_label}")

    # ── Load pose ─────────────────────────────────────────────────────────────
    unit   = proc_kwargs.get("signal_start_end_unit", "milliseconds")
    reduce = proc_kwargs.get("reduce_holistic_poses", True)

    print(f"\nPose file    : {args.pose_file}")
    print(f"Frames       : {args.signal_start} → {args.signal_end}  (unit: {unit})")
    print(f"Output dir   : {out}\n")

    original_pose, original_flat, body_shape, inv_norm = load_pose(
        args.pose_file, args.signal_start, args.signal_end,
        signal_start_end_unit=unit,
        reduce_holistic_poses=reduce,
        normalize=normalize,
    )
    T_orig = original_flat.shape[0]
    print(f"Original     : T={T_orig}, D={original_flat.shape[1]}")
    if inv_norm is not None:
        print(f"Inv-norm     : scale={inv_norm[0]:.2f}  cx={inv_norm[1]:.1f}  cy={inv_norm[2]:.1f}")

    stem = f"{Path(args.pose_file).stem}_{args.signal_start}_{args.signal_end}"

    orig_path = out / f"{stem}_original.pose"
    orig_to_save = tensor_to_pose(original_pose, original_flat, body_shape, inv_norm)
    save_pose(orig_to_save, orig_path)
    print(f"  Saved: {orig_path}  (pixel space)")

    # ── Generate augmented samples ────────────────────────────────────────────
    augmented_flats: List[Tuple[int, torch.Tensor]] = []

    for seed in range(args.n_samples):
        torch.manual_seed(seed)
        aug_flat = augment(original_flat, proc)
        augmented_flats.append((seed, aug_flat))

        aug_pose = tensor_to_pose(original_pose, aug_flat, body_shape, inv_norm)
        pose_path = out / f"{stem}_augmented_seed{seed}.pose"
        save_pose(aug_pose, pose_path)

        T_new = aug_flat.shape[0]
        ratio = T_orig / T_new
        print(f"  seed={seed}: T={T_orig} → T={T_new}  "
              f"({ratio:.2f}× {'faster' if ratio > 1 else 'slower' if ratio < 1 else 'same speed'})  "
              f"→ {pose_path.name}  (pixel space)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_rows = 1 + len(augmented_flats)
    x_max  = max(T_orig, *(t.shape[0] for _, t in augmented_flats))

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.8 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    plot_trajectory(axes[0], original_flat,
                    f"ORIGINAL  (T={T_orig})", "steelblue", x_max, coord_label)

    for ax, (seed, aug_flat) in zip(axes[1:], augmented_flats):
        T_new = aug_flat.shape[0]
        ratio = T_orig / T_new
        label = (f"augmented seed={seed}  |  {aug_names}  "
                 f"T={T_orig}→{T_new}  ({ratio:.2f}×)")
        plot_trajectory(ax, aug_flat, label, "darkorange", x_max, coord_label)

    coord_note = f"[augmentation in {coord_label} | .pose files in pixel space]"
    fig.suptitle(
        f"{Path(args.config).name}  ·  {Path(args.pose_file).name} "
        f"frames {args.signal_start}–{args.signal_end}  {coord_note}",
        fontsize=10,
    )
    fig.tight_layout()
    png_path = out / f"{stem}_augmentation_comparison.png"
    fig.savefig(png_path, dpi=150)
    print(f"\nFigure: {png_path}")


if __name__ == "__main__":
    main()
