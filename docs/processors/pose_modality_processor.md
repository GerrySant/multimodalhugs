# PoseModalityProcessor

`PoseModalityProcessor` loads pose sequences from `.pose` files, optionally reduces
the landmark set to a sign-language-relevant subset, and returns a `[T, D]` float
tensor per sample.

Requires the `pose-format` package:
```bash
pip install pose-format
# or
pip install "multimodalhugs[pose]"
```

---

## Input

The processor accepts:

| Signal type | What happens |
|---|---|
| `.pose` file path | Loaded via `pose_format.Pose.read()`. Clip window from `signal_start` / `signal_end` applied if non-zero. |
| `torch.Tensor` | Returned unchanged. |

The dataset column map must include `signal`, and may include `signal_start` and
`signal_end` for temporal clipping:

```yaml
column_map:
  signal: signal
  signal_start: signal_start
  signal_end: signal_end
```

---

## Output

| Stage | Shape |
|---|---|
| `process_sample` | `[T, D]` float32 — T frames, D landmarks×coordinates |
| `process_batch` | `[B, T_max, D]` float32, mask `[B, T_max]` |

---

## Landmark reduction (`reduce_holistic_poses`)

When `reduce_holistic_poses=True` (default), `pose_format.utils.generic.reduce_holistic`
is applied.  This collapses the full MediaPipe Holistic landmark set (face mesh, body,
hands, feet) into a smaller subset focused on signs: upper body, both hands, and a
compact face representation.  Leg landmarks are always hidden regardless of this flag.

Set `reduce_holistic_poses=False` to keep all landmarks from the `.pose` file.

---

## Clip window (`signal_start`, `signal_end`)

`signal_start` and `signal_end` are passed directly to `Pose.read()`.  The unit is
controlled by `signal_start_end_unit`:

| `signal_start_end_unit` | Interpretation |
|---|---|
| `SignalUnit.MILLISECONDS` (default) | Passed as `start_time` / `end_time` to `Pose.read()`. |
| `SignalUnit.FRAMES` | Passed as `start_frame` / `end_frame` to `Pose.read()`. |

When both `signal_start` and `signal_end` are `0`, the full file is loaded.

---

## YAML config example

```yaml
processor:
  slots:
    - processor_class: PoseModalityProcessor
      processor_kwargs:
        reduce_holistic_poses: true
        skip_frames_stride: 2
        signal_start_end_unit: milliseconds
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        signal: signal
        signal_start: signal_start
        signal_end: signal_end
```

Or using the `pipeline:` shorthand:

```yaml
processor:
  pipeline: pose2text
  tokenizer_path: facebook/m2m100_418M
  modality_kwargs:
    reduce_holistic_poses: true
    skip_frames_stride: 2
```

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reduce_holistic_poses` | `bool` | `True` | Apply `reduce_holistic` to keep only sign-relevant landmarks. |
| `skip_frames_stride` | `int \| None` | `None` | Keep every N-th frame; `None` keeps all. |
| `signal_start_end_unit` | `SignalUnit` | `SignalUnit.MILLISECONDS` | Unit for `signal_start` / `signal_end`. `MILLISECONDS` or `FRAMES`. |
