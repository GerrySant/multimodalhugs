# Pose Augmentations

`PoseModalityProcessor` supports a configurable augmentation pipeline applied to
pose sequences at training time. Augmentations are disabled at inference by default
and can be mixed freely in any order.

All augmentations operate on the **normalized** pose tensor — coordinates are in
shoulder-width units, centered on the shoulder midpoint — which is the same
representation the model receives. This ensures that augmentation values are
scale-invariant across datasets and signers.

---

## Configuration

Augmentations are declared in the `processor_kwargs` block of the
`PoseModalityProcessor` slot:

```yaml
processor:
  slots:
  - processor_class: PoseModalityProcessor
    processor_kwargs:
      # Comma-separated list of augmentations, applied in this order.
      data_augmentation_types: component_jitter,gaussian_noise

      # Per-augmentation probability (independent Bernoulli trials each sample).
      # Omit a key to always apply that augmentation (probability = 1.0).
      augmentation_probabilities:
        component_jitter: 0.5
        gaussian_noise: 0.8

      # Splits on which augmentation is active. Comma-separated or a list.
      augmentation_splits: train

      # Flat dict of keyword arguments forwarded to every augmentation function.
      # Unknown keys are silently ignored by each function.
      data_augmentation_kwargs:
        hand_jitter_std: 0.02
        noise_std: 0.005
```

### Key parameters

| Parameter | Type | Description |
|---|---|---|
| `data_augmentation_types` | `str` | Comma-separated augmentation names, applied left-to-right. |
| `augmentation_probabilities` | `dict` | Per-type probability in `[0, 1]`. Missing keys default to `1.0`. |
| `augmentation_splits` | `str` / `list` | Dataset split(s) where augmentation is active. Default: `train`. |
| `data_augmentation_kwargs` | `dict` | Flat kwargs forwarded to all augmentation functions. |
| `max_frames_after_augmentation` | `int` | Optional hard cap on sequence length after augmentation, applied by random cropping. |

---

## Available Augmentations

### 1. `fixed_speed_factor`

Resample the entire sequence to a fixed speed. A factor greater than 1 produces
fewer frames (faster signing); less than 1 produces more frames (slower).

```yaml
data_augmentation_types: fixed_speed_factor
data_augmentation_kwargs:
  speed_factor: 2.0   # 2× faster → half the frames
```

| Parameter | Default | Description |
|---|---|---|
| `speed_factor` | `1.0` | Divisor applied to the frame count. Must be > 0. |

**Use case:** simulate a specific known speed ratio (e.g. back-translating at 2×
and training the SLT model to match real-data speed).

---

### 2. `random_speed_perturbation`

Split the sequence into randomly placed segments and resample each one
independently with a uniformly sampled speed factor. This simulates the natural
local speed variation that occurs within a single signing clip.

```yaml
data_augmentation_types: random_speed_perturbation
data_augmentation_kwargs:
  min_factor: 0.8
  max_factor: 3.0
  min_num_segments: 3
  max_num_segments: 7
```

| Parameter | Default | Description |
|---|---|---|
| `min_factor` | `1.0` | Lower bound for per-segment speed factor. |
| `max_factor` | `6.0` | Upper bound for per-segment speed factor. |
| `num_segments` | `5` | Fixed number of segments (used when neither `min_num_segments` nor `max_num_segments` is set). |
| `min_num_segments` | — | Lower bound for a randomly drawn segment count. |
| `max_num_segments` | — | Upper bound for a randomly drawn segment count. When set, `num_segments` is ignored. |

**Use case:** data augmentation for SLT when the training set contains sequences
at a single speed (e.g. back-translated data always at 3×) — randomising speed
locally prevents the model from over-fitting to that artefact.

---

### 3. `component_jitter`

Add a smooth spatial displacement to the hands across the sequence. The
displacement is sampled independently for the start and end of the clip and
linearly interpolated across frames, so different signs within the clip are
performed at slightly varying positions rather than a single fixed offset.

```yaml
data_augmentation_types: component_jitter
data_augmentation_kwargs:
  global_jitter_std: 0.0      # recommended: keep at 0 (see note below)
  hand_jitter_std: 0.02
  elbow_follow_fraction: 0.3
  hands_move_together: true
```

| Parameter | Default | Description |
|---|---|---|
| `global_jitter_std` | `0.02` | Std of a single offset added to every keypoint. Recommended value: `0` — after normalization all sequences are already centered on the shoulder midpoint, so a global shift creates variation that does not exist in real data. |
| `hand_jitter_std` | `0.01` | Std of the per-hand displacement in shoulder-width units. Each hand gets independent start/end offsets interpolated across T frames. |
| `elbow_follow_fraction` | `0.0` | How much the elbow follows the wrist displacement (`0` = elbow fixed, `1` = full follow). Only the component **perpendicular** to the upper-arm axis is applied so the elbow rotates rather than stretching the arm. |
| `hands_move_together` | `True` | If `True`, both hands share the same displacement trajectory. If `False`, each hand drifts independently. |

**Coordinate scale:** `hand_jitter_std=0.01` corresponds to ~1% of a
shoulder-width. Empirically, values in `[0.01, 0.05]` produce natural-looking
variation.

---

### 4. `gaussian_noise`

Add independent Gaussian noise to every keypoint coordinate at every frame.
Unlike `component_jitter`, the noise is uncorrelated across frames, which
prevents the model from memorising exact coordinate sequences.

```yaml
data_augmentation_types: gaussian_noise
data_augmentation_kwargs:
  noise_std: 0.01
  noise_mean: 0.0
```

| Parameter | Default | Description |
|---|---|---|
| `noise_std` | `0.01` | Standard deviation of the noise in shoulder-width units. |
| `noise_mean` | `0.0` | Mean of the noise distribution. |

**Use case:** a lightweight regulariser to prevent overfitting on small datasets.
Can be combined freely with other augmentations.

---

### 5. `appearance_transfer`

Transfer the body appearance (torso and limb proportions) of a randomly selected
signer onto the input sequence while preserving the hand shapes and locations that
encode the actual sign content.

The appearance is sampled from a pre-built `.pose` file containing resting-posture
frames from multiple signers and datasets (built with `build_appearances_pose.py`).

```yaml
data_augmentation_types: appearance_transfer
augmentation_probabilities:
  appearance_transfer: 0.5
data_augmentation_kwargs:
  appearance_pose_path: /path/to/apperances.pose
  cache_appearances: true
```

| Parameter | Default | Description |
|---|---|---|
| `appearance_pose_path` | — | **Required.** Path to the appearances `.pose` file. Each frame should be a different signer at rest, preprocessed with `reduce_holistic` (178 keypoints). |
| `cache_appearances` | `True` | If `True`, the appearance frames are loaded from disk once and kept in memory for the lifetime of the process. Set to `False` if memory is constrained or if the file may change between runs. |

#### How it works

1. Each frame in the appearances file is normalized to shoulder-width units at
   load time (same convention as `PoseModalityProcessor`).
2. One frame is selected at random and used as the **target appearance**.
3. The constant delta `target_appearance − x[0]` is added to every frame of the
   input sequence, shifting the signer's body posture to match the target.
4. Hand keypoints (`LEFT_HAND`, `RIGHT_HAND`) and POSE wrist keypoints are
   restored from the original sequence — they carry the sign content and must
   not be overwritten.
5. The result is renormalized per frame (shoulder distance = 1.0, shoulder
   midpoint at origin) to restore the invariant that may be broken by the shift.

#### Building the appearances file

Use `scripts/check_speed_factor/build_appearances_pose.py` to build the file.
It samples resting-posture frames from one or more TSV datasets, filters out
signing frames (wrists above elbows), applies `pose_hide_legs` + `reduce_holistic`,
and writes the result as a single multi-frame `.pose` file.

---

## Combining augmentations

Augmentations listed in `data_augmentation_types` are applied **sequentially**,
each independently gated by its probability. Example combining all five:

```yaml
data_augmentation_types: random_speed_perturbation,component_jitter,gaussian_noise,appearance_transfer
augmentation_probabilities:
  random_speed_perturbation: 1.0
  component_jitter: 0.7
  gaussian_noise: 0.5
  appearance_transfer: 0.5
augmentation_splits: train
data_augmentation_kwargs:
  # random_speed_perturbation
  min_factor: 0.8
  max_factor: 3.0
  min_num_segments: 3
  max_num_segments: 7
  # component_jitter
  global_jitter_std: 0.0
  hand_jitter_std: 0.02
  elbow_follow_fraction: 0.3
  hands_move_together: true
  # gaussian_noise
  noise_std: 0.005
  # appearance_transfer
  appearance_pose_path: /path/to/apperances.pose
  cache_appearances: true
```

---

## Visualising augmentations

The debug script `debug_run_scripts/debug/visualize_augmentation.py` renders
augmented samples from any training config to `.pose` files and a comparison
plot, making it easy to inspect the effect before training:

```bash
conda activate bt-slt-aug

python debug_run_scripts/debug/visualize_augmentation.py \
    --config debug_run_scripts/debug/augmentation_appearance.yaml \
    --pose_file /path/to/sample.pose \
    --signal_start 1679 --signal_end 1875 \
    --out_dir /tmp/aug_debug \
    --n_samples 3
```

The script saves `<stem>_original.pose`, `<stem>_augmented_seed{n}.pose`, and a
trajectory comparison PNG, all in pixel coordinates so they open correctly in any
pose viewer.
