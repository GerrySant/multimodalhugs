# FeaturesModalityProcessor

`FeaturesModalityProcessor` loads precomputed feature sequences from `.npy` files and
returns a `[T, D]` float tensor per sample.  Use this when features (e.g. frame-level
embeddings) have been extracted offline and stored as numpy arrays.

No optional dependency is required.

---

## Input

| Signal type | What happens |
|---|---|
| `.npy` file path | Loaded via `numpy.load()`. Temporal axis moved to position 0 if needed. Frame skipping applied if configured. |
| `torch.Tensor` | Returned unchanged. |
| `np.ndarray` | Converted to a float32 tensor. |

---

## Output

| Stage | Shape |
|---|---|
| `process_sample` | `[T, D]` float32 |
| `process_batch` | `[B, T_max, D]` float32, mask `[B, T_max]` |

---

## Temporal axis position (`temporal_dimension_position`)

`.npy` files may store feature arrays with the temporal axis at any position.  The
processor uses `torch.movedim` to move it to position 0 so that the output is always
`[T, D]`.

- `temporal_dimension_position=0` (default): no permutation.
- `temporal_dimension_position=1`: swaps axes 0 and 1 (e.g. `[D, T]` → `[T, D]`).

---

## LRU cache (`use_cache`)

When `use_cache=True` (default), `_load_from_disk` is wrapped with an LRU cache.
Cache size is derived automatically from available system memory (or SLURM allocation),
assuming approximately 0.7 MB per cached feature file.

Caching is useful during overfitting experiments where the same files are read many
times.  Disable with `use_cache=False` in production training runs where the dataset
is large.

---

## YAML config example

```yaml
processor:
  slots:
    - processor_class: FeaturesModalityProcessor
      processor_kwargs:
        skip_frames_stride: 2
        temporal_dimension_position: 0
        use_cache: false
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
  pipeline: features2text
  tokenizer_path: facebook/m2m100_418M
  modality_kwargs:
    skip_frames_stride: 2
    use_cache: false
```

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skip_frames_stride` | `int \| None` | `None` | Keep every N-th frame; `None` keeps all. |
| `temporal_dimension_position` | `int` | `0` | Index of the temporal axis in the raw `.npy` array. Moved to position 0 via `torch.movedim`. |
| `use_cache` | `bool` | `True` | Cache loaded files with `lru_cache`. Size derived from available memory. |
