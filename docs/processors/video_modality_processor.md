# VideoModalityProcessor

`VideoModalityProcessor` loads video clips from disk (or HTTP/HTTPS URLs), extracts
frames, and optionally preprocesses them through a HuggingFace image processor.
Video decoding is delegated to `transformers.video_utils.load_video()`, which
supports five backends.

---

## Backends

The `backend` parameter selects the decoding library.  Backend availability is
deferred to call time — `VideoModalityProcessor` initialises successfully even if the
chosen backend is not installed; the `ImportError` is raised only when the first
video is decoded.

| `backend` | Decoder | Notes |
|---|---|---|
| `"pyav"` (default) | CPU | Widely available, handles most formats. Safe with any `dataloader_num_workers`. |
| `"torchvision"` | CPU | PyTorch-native. |
| `"decord"` | CPU | Fast random access. |
| `"opencv"` | CPU | No audio support. |
| `"torchcodec"` | CPU (default) or **GPU** | Hardware codec decode. CPU by default; set `device: "cuda"` to decode directly to CUDA tensors via NVDEC, eliminating the CPU→GPU copy. Requires `worker_start_method: spawn` when `dataloader_num_workers > 0` on Linux — see the torchcodec section below. |

---

## Frame sampling

Two mutually exclusive parameters control how many frames are extracted from a clip
window:

- `num_frames=N` — uniformly subsample exactly **N** frames via
  `np.linspace(start, end-1, N)`.  Output always has exactly N frames regardless of
  clip length.  Use when the model expects a fixed temporal dimension.
- `skip_frames_stride=N` — keep every **N**-th frame via
  `np.arange(start, end, N)`.  Output length scales with clip length (a 50-frame clip
  with stride 2 gives 25 frames; a 100-frame clip gives 50).  Use when variable-length
  sequences are acceptable and you want proportional frame-rate reduction.

When both are set, `num_frames` takes precedence **only when `num_frames < clip_length`**.
If the clip is shorter than `num_frames` (e.g. a 10-frame clip with `num_frames=16`),
the condition is false and execution falls through to `skip_frames_stride` if set, or
all frames otherwise.  Setting both is redundant in the common case — pick one.

---

## Clip window (`signal_start`, `signal_end`)

The dataset columns `signal_start` and `signal_end` define the clip window within a
video file.  The unit is controlled by `signal_start_end_unit`:

| `signal_start_end_unit` | Interpretation |
|---|---|
| `SignalUnit.MILLISECONDS` (default) | Values in milliseconds; converted to frame indices using the video's FPS. |
| `SignalUnit.FRAMES` | Values are used directly as frame indices. |

When both `signal_start` and `signal_end` are `0`, the full video is loaded regardless
of the unit setting.

---

## Output format

### Without `custom_preprocessor_path` (default)

Frames are returned as a raw float32 tensor of shape `[T, C, H, W]` with pixel values
in the 0–255 range.  No resizing or normalisation is applied.

The only available post-decode transformation is `join_chw=True`, which merges
C, H, W into a single feature axis, producing `[T, C*H*W]` — useful when the model
expects a flat feature vector per frame.

### With `custom_preprocessor_path`

Frames are passed through the specified HuggingFace image processor (e.g.
`"openai/clip-vit-base-patch32"`) after decoding.  The processor handles resizing,
normalisation, and channel reordering, returning `[T, C, H, W]` in the model's
expected pixel-value range.

`process_batch` pads along T and returns `[B, T_max, C, H, W]` with a `[B, T_max]`
mask for both output modes.

---

## torchcodec and GPU decode

`backend="torchcodec"` uses hardware codec infrastructure.  By default it decodes to
CPU.  Set `device="cuda"` (or `"cuda:N"`) to activate NVDEC — a dedicated hardware
video decode engine on NVIDIA GPUs that runs independently of CUDA compute cores.
Decoded frames arrive as CUDA tensors without any CPU→GPU copy.

With transformers 5.x, the default `TorchvisionBackend` image processor accepts
tensors directly and runs on the same device as the input.  When
`backend="torchcodec"`, `device="cuda"`, and `custom_preprocessor_path` is set,
the full pipeline (decode + resize + normalise) runs on GPU with zero CPU involvement.

**Worker start method constraint (Linux only)**

On Linux, PyTorch DataLoader defaults to the `"fork"` start method.  A process that
has an active CUDA context cannot safely be forked — worker processes inherit a
broken state, causing errors or deadlocks when `dataloader_num_workers > 0`.

`VideoModalityProcessor` emits a `logger.warning` at construction time when
`device="cuda"` is combined with the `"fork"` start method.

To use GPU decode safely with multiple workers, set `worker_start_method: spawn` in
the training config:

```yaml
training:
  dataloader_num_workers: 4
  worker_start_method: spawn
```

Or call `torch.multiprocessing.set_start_method("spawn")` before training, or set
`dataloader_num_workers: 0`.

---

## YAML config examples

### Minimal (pyav backend, no preprocessing)

```yaml
processor:
  slots:
    - processor_class: VideoModalityProcessor
      processor_kwargs:
        skip_frames_stride: 2
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        signal: signal
        signal_start: signal_start
        signal_end: signal_end
```

### With CLIP preprocessing (CPU decode)

```yaml
processor:
  slots:
    - processor_class: VideoModalityProcessor
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        backend: pyav
        skip_frames_stride: 2
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        signal: signal
        signal_start: signal_start
        signal_end: signal_end
```

### With CLIP preprocessing (GPU decode via torchcodec)

```yaml
processor:
  slots:
    - processor_class: VideoModalityProcessor
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        backend: torchcodec
        device: "cuda"
        num_frames: 16
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        signal: signal
        signal_start: signal_start
        signal_end: signal_end

training:
  worker_start_method: spawn   # required for GPU decode with num_workers > 0
```

Or using the `pipeline:` shorthand with `modality_kwargs`:

```yaml
processor:
  pipeline: video2text
  tokenizer_path: facebook/m2m100_418M
  modality_kwargs:
    custom_preprocessor_path: openai/clip-vit-base-patch32
    backend: torchcodec
    device: "cuda"
    num_frames: 16
```

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_preprocessor_path` | `str \| None` | `None` | HuggingFace model ID or local path to an image processor. When set, decoded frames are passed through the processor after decoding. |
| `backend` | `str` | `"pyav"` | Decoding backend. One of `"pyav"`, `"torchvision"`, `"decord"`, `"opencv"`, `"torchcodec"`. |
| `device` | `str \| None` | `None` | Decode device for `torchcodec` only. `None` decodes to CPU. `"cuda"` or `"cuda:N"` activates NVDEC. Ignored for other backends. |
| `num_frames` | `int \| None` | `None` | Uniformly subsample this many frames. Takes precedence over `skip_frames_stride`. |
| `skip_frames_stride` | `int \| None` | `None` | Keep every N-th frame. Ignored when `num_frames` is set. |
| `join_chw` | `bool` | `False` | Merge C, H, W into a single feature dimension: `[T, C*H*W]`. Only used without `custom_preprocessor_path`. |
| `use_cache` | `bool` | `False` | Cache decoded clips with `lru_cache`. Useful for repeated access to the same files (e.g. overfitting experiments). Cache size is set dynamically based on available memory. |
| `io_max_retries` | `int` | `3` | Retry count on transient I/O errors, with exponential backoff (1 s, 2 s, 4 s). |
| `signal_start_end_unit` | `SignalUnit` | `SignalUnit.MILLISECONDS` | Unit for `signal_start` / `signal_end` in the dataset. `MILLISECONDS` or `FRAMES`. |
