# Processor Architecture

## Overview

The processor layer sits between raw dataset items and the model forward pass. It is responsible for loading files from disk, converting signals into tensors, padding variable-length sequences, and producing the exact dictionary of named tensors that the model expects.

The design is built around three composable layers:

```
ModalityProcessor        — knows one modality (pose, video, text, image, …)
ProcessorSlot            — binds a ModalityProcessor to a TSV column and a forward() key
MultimodalMetaProcessor  — orchestrates all slots; produces the full model batch
```

---

## Layer 1: `ModalityProcessor`

`ModalityProcessor` (`multimodalhugs/processors/modality_processor.py`) is the base class for all modality-specific processing logic. It has no knowledge of task structure — it does not know whether it is processing encoder input, labels, or prompts. That context belongs to the `ProcessorSlot`.

### Interface

```python
class ModalityProcessor(ABC):

    def process_sample(self, values, **kwargs) -> Any:
        """
        Load and preprocess a single sample.
        Called at dataset.with_transform() time — before batching.

        values is either:
          - a raw value (file path, string, ndarray, tensor, …) for single-column slots
          - a dict {processor_param_name: value} for multi-column slots, e.g.:
              {"signal": "/data/sample.pose", "signal_start": 0, "signal_end": 500}

        The default implementation is a no-op (returns values unchanged).
        Override this when per-sample preprocessing is needed (file decoding,
        resizing, format conversion, etc.).
        """

    @abstractmethod
    def process_batch(self, samples: List[Any], **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Pad a list of pre-loaded values into a batch tensor and an optional mask.
        Called inside the DataCollator after the full batch is assembled.

        Returns:
            (data_tensor, mask_tensor)
            mask_tensor may be None when no padding mask is meaningful (e.g. labels).
        """
```

### Two-stage processing

The split between `process_sample` and `process_batch` mirrors the two stages in the HuggingFace data pipeline:

| Stage | When | What happens |
|---|---|---|
| `process_sample` | `dataset.with_transform()` time, per item | Decode file, convert to tensor. No padding. |
| `process_batch` | Inside `DataCollator.__call__`, per batch | Pad to common length, create mask. |

This means expensive I/O (reading video or pose files) is done lazily per item during DataLoader prefetch, while padding — which requires knowing the full batch — is done at collation time.

### Built-in modality processors

| Class | Modality | Key parameters |
|---|---|---|
| `PoseModalityProcessor` | `.pose` files | `reduce_holistic_poses`, `skip_frames_stride` |
| `VideoModalityProcessor` | Video files | `skip_frames_stride`, `join_chw`, `use_cache` |
| `ImageModalityProcessor` | Image files / text-rendered images | `font_path`, `width`, `height`, `normalize_image`, `mean`, `std` |
| `FeaturesModalityProcessor` | `.npy` / `.pt` feature files | `skip_frames_stride`, `temporal_dimention_position`, `use_cache` |
| `SignwritingModalityProcessor` | FSW SignWriting strings | `custom_preprocessor_path`, `width`, `height`, `channels` |
| `TextModalityProcessor` | Text strings | `tokenizer`, `tokenizer_path`, `new_vocabulary`, `role` (`TextRole.INPUT` or `TextRole.TARGET`) |

`TextModalityProcessor` is the only processor that carries a tokenizer. The `role` parameter (a `TextRole` enum) controls how the batch is assembled:

- `TextRole.INPUT` — receives a list of strings; returns `(token_ids [B, L], attention_mask [B, L])`. Used for both encoder prompts and decoder prompts.
- `TextRole.TARGET` — receives a list of dicts `{"target_prefix": ..., "target": ...}`; concatenates them, appends EOS, pads with `-100`; returns `(labels [B, L], None)`.

`new_vocabulary` is an optional path to a vocabulary file. When provided, the processor extends the tokenizer internally and exposes `self.new_tokens` (the added tokens) and `self.pretrained_tokenizer` (the unextended copy) as bridge attributes for setup scripts.

---

## Layer 2: `ProcessorSlot`

`ProcessorSlot` (`multimodalhugs/processors/meta_processor.py`) is a dataclass that binds a `ModalityProcessor` to:

1. **which dataset columns to read** — via `column_map`
2. **what keys to write in the output batch** — via `output_data_key` and `output_mask_key`
3. **whether it produces a loss target** — via `is_label`

```python
@dataclass
class ProcessorSlot:
    processor: ModalityProcessor
    output_data_key: str                         # key for the data tensor in the batch dict
    output_mask_key: Optional[str] = None        # key for the mask tensor (None = no mask)
    column_map: Dict[str, str] = field(
        default_factory=lambda: {"signal": "signal"}
    )
    is_label: bool = False                       # marks this slot as a loss target
```

### `column_map` in depth

`column_map` maps **dataset item field names** (TSV column names) to **processor parameter names** (the keys inside the dict passed to `process_sample`).

- The **first key** is the *primary field*. Its value is replaced with a preprocessed tensor by `_transform_get_items_output` when `process_sample` returns a tensor.
- All **subsequent keys** are context-only: they are passed to `process_sample` but not written back into the dataset item. Typical use: temporal bounds (`signal_start`, `signal_end`).

**Default:** `{"signal": "signal"}` — reads the `signal` TSV column and passes it as `signal` to `process_sample`. This covers the standard single-field case.

**Multi-column example (pose with temporal bounds):**
```python
column_map={
    "signal": "signal",               # primary field → written back as tensor
    "signal_start": "signal_start",   # context only
    "signal_end":   "signal_end",     # context only
}
```

**Label slot — two required columns:**
```python
# TSV columns "decoder_prompt" and "output" map to processor params "target_prefix" and "target"
column_map={"decoder_prompt": "target_prefix", "output": "target"}
```

**Non-standard column name (multi-input scenario):**
```python
# When two encoder streams both need temporal bounds, field names must differ
column_map={
    "pose_signal":       "signal",
    "pose_signal_start": "signal_start",
    "pose_signal_end":   "signal_end",
}
```

### `is_label`

Marks a slot as producing a loss target. Does not affect processing logic inside the `MultimodalMetaProcessor` — the processor iterates all slots identically. It is an annotation for callers (trainers, collators) to identify which output key carries the target sequence without hardcoding the key name.

---

## Layer 3: `MultimodalMetaProcessor`

`MultimodalMetaProcessor` (`multimodalhugs/processors/meta_processor.py`) takes a flat list of `ProcessorSlot` objects and produces a complete `BatchFeature`.

```python
class MultimodalMetaProcessor(ProcessorMixin):
    def __init__(
        self,
        slots: List[ProcessorSlot],   # all slots in processing order
        tokenizer = None,             # kept for HF ProcessorMixin compatibility
    ): ...
```

The meta processor has **no knowledge of task structure** — it does not distinguish encoder inputs from labels from prompts. All semantic meaning lives in the processors and their slot configuration.

### `_transform_get_items_output` — dataset-level hook

Called via `dataset.with_transform(processor._transform_get_items_output)`. Iterates over all slots and calls `slot.processor.process_sample()` on each item's primary field. Only writes the result back when `process_sample` returns a tensor — text slots (which are no-ops) do not corrupt their columns.

```python
dataset = dataset.with_transform(meta_processor._transform_get_items_output)
```

### `__call__` — collator-level call

Called by `DataCollatorMultimodalSeq2Seq` with a list of sample dicts. Iterates slots in declaration order. Skips a slot if its `output_data_key` is already present in the result — this allows callers to pre-populate keys (e.g. `decoder_input_ids` derived from labels by the model).

### Save and load

`MultimodalMetaProcessor` overrides the HF save/load interface. Each slot is serialised to `processor_config.json` with its processor class name, constructor kwargs, output keys, column map, and `is_label` flag.

```python
meta.save_pretrained("/path/to/save/")
loaded = MultimodalMetaProcessor.from_pretrained("/path/to/save/")
```

---

## Built-in task processors (legacy wrappers)

The six task-specific processors are direct subclasses of `MultimodalMetaProcessor` in `multimodalhugs/processors/legacy/`. They wire up the standard four-slot configuration for each modality and are fully backward-compatible.

| Class | Encoder modality | `output_data_key` |
|---|---|---|
| `Pose2TextTranslationProcessor` | `PoseModalityProcessor` | `input_frames` |
| `Video2TextTranslationProcessor` | `VideoModalityProcessor` | `input_frames` |
| `Image2TextTranslationProcessor` | `ImageModalityProcessor` | `input_frames` |
| `Features2TextTranslationProcessor` | `FeaturesModalityProcessor` | `input_frames` |
| `SignwritingProcessor` | `SignwritingModalityProcessor` | `input_frames` |
| `Text2TextTranslationProcessor` | `TextModalityProcessor` | `input_ids` |

All six are re-exported from `multimodalhugs.processors` so existing import paths are unchanged.

---

## Integration with `DataCollatorMultimodalSeq2Seq`

When the processor is a `MultimodalMetaProcessor` (which includes all legacy subclasses), the collator delegates all work to it:

```python
def __call__(self, samples):
    batch = self.processor(samples)
    if (
        "labels" in batch
        and self.model is not None
        and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        and self.model.training
    ):
        batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
            labels=batch["labels"]
        )
    return batch
```

The collator no longer needs a tokenizer — label processing lives inside `TextModalityProcessor(role=TextRole.TARGET)`.

---

## Data flow summary

```
TSV file
  └── GeneratorBasedBuilder._generate_examples()
        └── HuggingFace Dataset (raw dicts)
              └── dataset.with_transform(meta._transform_get_items_output)
                    │  process_sample() per item — file I/O, tensor conversion
                    └── DataLoader (batches of dicts, primary fields are tensors)
                          └── DataCollatorMultimodalSeq2Seq.__call__()
                                └── MultimodalMetaProcessor.__call__()
                                      │  process_batch() per slot — padding, masking
                                      └── BatchFeature → model.forward()
```
