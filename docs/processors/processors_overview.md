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
| `TextModalityProcessor` | Text strings | `tokenizer`, `role` (`"encoder"`, `"prompt"`, `"label"`) |

`TextModalityProcessor` is the only processor that carries a tokenizer. The `role` parameter controls how the batch is assembled:

- `"encoder"` / `"prompt"` — receives a list of strings; returns `(token_ids [B, L], attention_mask [B, L])`.
- `"label"` — receives a list of full sample dicts (needs both `decoder_prompt` and `output`); concatenates them, appends EOS, pads with `-100`; returns `(labels [B, L], None)`.

---

## Layer 2: `ProcessorSlot`

`ProcessorSlot` (`multimodalhugs/processors/meta_processor.py`) is a dataclass that binds a `ModalityProcessor` to:

1. **which dataset columns to read** — via `column_map`
2. **what keys to write in the output batch** — via `output_data_key` and `output_mask_key`

```python
@dataclass
class ProcessorSlot:
    processor: ModalityProcessor
    output_data_key: str                         # key for the data tensor in the batch dict
    output_mask_key: Optional[str] = None        # key for the mask tensor (None = no mask)
    column_map: Dict[str, str] = field(
        default_factory=lambda: {"signal": "signal"}
    )
```

### `column_map` in depth

`column_map` maps **dataset item field names** (TSV column names) to **processor parameter names** (the keys inside the dict passed to `process_sample`).

- The **first key** is the *primary field*. Its value is replaced with a preprocessed tensor by `_transform_get_items_output`. This is the main signal column.
- All **subsequent keys** are context-only: they are passed to `process_sample` but not written back into the dataset item. Typical use: temporal bounds (`signal_start`, `signal_end`).

**Default:** `{"signal": "signal"}` — reads the `signal` TSV column and passes it as `signal` to `process_sample`. This covers the standard single-field case and requires no explicit configuration.

**Multi-column example (pose with temporal bounds):**
```python
column_map={
    "signal": "signal",          # primary field → written back as tensor
    "signal_start": "signal_start",   # context only
    "signal_end":   "signal_end",     # context only
}
```

**Non-standard column name example (multi-input scenario):**
```python
# When two encoder streams both need temporal bounds, field names must differ
column_map={
    "pose_signal":       "signal",       # dataset column → processor param
    "pose_signal_start": "signal_start",
    "pose_signal_end":   "signal_end",
}
```

**Prompt / label slots use a remapped primary field:**
```python
# Encoder prompt: reads the "encoder_prompt" TSV column
column_map={"encoder_prompt": "signal"}

# Decoder prompt: reads the "decoder_prompt" TSV column
column_map={"decoder_prompt": "signal"}
```
For the label slot the `column_map` is ignored at collation time — the full sample dict is always passed to `process_batch` because label construction needs both `decoder_prompt` and `output`.

---

## Layer 3: `MultimodalMetaProcessor`

`MultimodalMetaProcessor` (`multimodalhugs/processors/meta_processor.py`) orchestrates all slots and produces a complete `BatchFeature` that is ready for `model.forward()`.

```python
class MultimodalMetaProcessor(ProcessorMixin):
    def __init__(
        self,
        encoder_slots: List[ProcessorSlot],        # one per encoder input stream
        label_slot: ProcessorSlot,                 # target sequence
        encoder_prompt_slot: Optional[ProcessorSlot] = None,
        decoder_prompt_slot: Optional[ProcessorSlot] = None,
        tokenizer = None,                          # kept for HF ProcessorMixin compatibility
    ): ...
```

### `_transform_get_items_output` — dataset-level hook

Called via `dataset.with_transform(processor._transform_get_items_output)`. Iterates over all slots and calls `slot.processor.process_sample()` on each item's primary field, converting raw values (file paths, strings) to tensors in-place. Non-primary fields (temporal bounds, etc.) are passed to `process_sample` but not modified.

```python
dataset = dataset.with_transform(meta_processor._transform_get_items_output)
```

This runs in the DataLoader worker processes, so file I/O is parallelised automatically.

### `__call__` — collator-level call

Called by `DataCollatorMultimodalSeq2Seq` with a list of sample dicts (one dict per batch item). Each sample's primary field is already a tensor at this point (converted by `_transform_get_items_output`).

Processing order:
1. Encoder slots — `process_batch` on each slot; writes `output_data_key` and `output_mask_key`.
2. Encoder prompt slot — same pattern.
3. Decoder prompt slot — same pattern (skipped if key already populated by the collator).
4. Label slot — `process_batch` receives the full list of sample dicts; writes `output_data_key`.

Returns a `BatchFeature` (a dict-like HF object) ready for the model.

### Save and load

`MultimodalMetaProcessor` overrides the HF `save_pretrained` / `from_pretrained` interface to handle the dynamic slot composition that `ProcessorMixin`'s static `attributes` list cannot express.

**Saving** serialises each slot to JSON (`processor_config.json`) with:
- `processor_class` — the class name string
- `processor_kwargs` — JSON-serialisable `__dict__` values (private attrs and tokenizer excluded)
- `output_data_key`, `output_mask_key`, `column_map`

The tokenizer is saved alongside (standard HF behaviour).

**Loading** reads `processor_config.json`, imports each processor class from `multimodalhugs.processors`, reconstructs it with the saved kwargs, and rebuilds the `ProcessorSlot` list.

```python
# Save
meta.save_pretrained("/path/to/save/")

# Load
meta = MultimodalMetaProcessor.from_pretrained("/path/to/save/")
# or via AutoProcessor if registered:
meta = AutoProcessor.from_pretrained("/path/to/save/")
```

---

## Built-in task processors (legacy wrappers)

The six task-specific processors from previous versions are now direct subclasses of `MultimodalMetaProcessor` in `multimodalhugs/processors/legacy/`. They wire up the standard slot configuration for each modality and are fully backward-compatible.

| Class | Location | Encoder modality | `output_data_key` |
|---|---|---|---|
| `Pose2TextTranslationProcessor` | `legacy/pose2text_preprocessor.py` | `PoseModalityProcessor` | `input_frames` |
| `Video2TextTranslationProcessor` | `legacy/video2text_preprocessor.py` | `VideoModalityProcessor` | `input_frames` |
| `Image2TextTranslationProcessor` | `legacy/image2text_preprocessor.py` | `ImageModalityProcessor` | `input_frames` |
| `Features2TextTranslationProcessor` | `legacy/features2text_preprocessor.py` | `FeaturesModalityProcessor` | `input_frames` |
| `SignwritingProcessor` | `legacy/signwriting_preprocessor.py` | `SignwritingModalityProcessor` | `input_frames` |
| `Text2TextTranslationProcessor` | `legacy/text2text_preprocessor.py` | `TextModalityProcessor` | `input_ids` |

All six are re-exported from `multimodalhugs.processors` so existing import paths are unchanged.

---

## Integration with `DataCollatorMultimodalSeq2Seq`

When the processor is a `MultimodalMetaProcessor` (which includes all legacy subclasses), the collator delegates all work to the processor:

```python
def __call__(self, samples):
    batch = self.processor(samples)         # all processing happens here
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

The collator no longer needs a tokenizer — label processing lives inside `TextModalityProcessor(role="label")` in the `label_slot`.

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
