# Processor Redesign: Modality-First Architecture

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Implement `ModalityProcessor` base + concrete modality processors | ✅ Done |
| 2 | Implement `ProcessorSlot` dataclass | ✅ Done |
| 3 | Implement `MultimodalMetaProcessor` with HF save/load | ✅ Done |
| 4 | Move label processing into `TextModalityProcessor(role=TextRole.TARGET)` | ✅ Done |
| 5 | Simplify `DataCollatorMultimodalSeq2Seq` | ✅ Done |
| 6 | Update `training_setup/` scripts to use flat `slots=[...]` | ✅ Done |
| 7 | Wrap legacy task-specific processors as thin `MultimodalMetaProcessor` subclasses | ✅ Done |
| 8 | Add `build_processor_from_config` — declarative slot builder for `multimodalhugs-setup` | ✅ Done |
| 9 | Add shorthand processor config format for common use cases | 🔲 Todo |
| 10 | Unified `multimodalhugs-setup` CLI — single general setup command | 🔲 Todo |
| 11 | Update dataset TSV handling for multi-column inputs | Deferred (#71) |
| 12 | Extend `MultiModalEmbedderModel.forward()` for multi-stream input | Deferred (#72) |

---

## Motivation

The original processor design (`Pose2TextTranslationProcessor`, `Video2TextTranslationProcessor`, etc.) coupled three separate concerns into a single class:

1. **What data to load** — modality-specific file loading and feature extraction
2. **How to combine inputs** — orchestrating encoder inputs, prompts, and labels
3. **What output format to produce** — always text, always tokenized in the DataCollator

This made it impossible to:
- Reuse modality logic across tasks (e.g., `PoseModalityProcessor` shared by `pose2text` and `pose2pose`)
- Handle multiple encoder inputs (`video + pose → text`)
- Handle non-text outputs (`text + image → pose`)

---

## Implemented Architecture

Three composable layers replace the task-specific processors:

```
ModalityProcessor        — knows one modality (pose, video, text, image, …)
ProcessorSlot            — binds a ModalityProcessor to TSV columns and a forward() key
MultimodalMetaProcessor  — orchestrates a flat list of slots; produces the full model batch
```

### Key design decision: flat `slots` list

An early draft proposed named constructor parameters (`encoder_slots`, `label_slot`, `encoder_prompt_slot`, `decoder_prompt_slot`) on `MultimodalMetaProcessor`. These were dropped in favour of a single flat `slots: List[ProcessorSlot]` for two reasons:

1. Named params re-introduce task-structure assumptions into the meta-processor, which was the core problem being solved.
2. A flat list is strictly more general: any number of encoder streams, any modality as a label, optional prompt slots — all expressed uniformly without special-casing.

Task semantics are expressed entirely through `ProcessorSlot` configuration (`output_data_key`, `column_map`, `is_label`), not through the meta-processor's constructor.

---

## Layer 1: `ModalityProcessor`

A pure modality-specific class with no knowledge of task structure.

```python
class ModalityProcessor(ABC):

    def process_sample(self, values, **kwargs) -> Any:
        """
        Load and preprocess a single sample.
        Called at dataset.with_transform() time — no padding, no batching.

        values is either:
          - a raw value (file path, string, tensor, …) for single-column slots
          - a dict {processor_param_name: value} for multi-column slots

        Default implementation is a no-op (returns values unchanged).
        """
        return values

    @abstractmethod
    def process_batch(self, samples: List[Any], **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Pad a list of pre-loaded values into a batch tensor and an optional mask.
        Called inside the DataCollator after the full batch is assembled.
        Returns (data_tensor, mask_tensor).  mask_tensor may be None.
        """
```

**Concrete implementations:**

| Class | Modality | Key parameters |
|---|---|---|
| `PoseModalityProcessor` | `.pose` files | `reduce_holistic_poses`, `skip_frames_stride` |
| `VideoModalityProcessor` | Video files | `skip_frames_stride`, `join_chw`, `use_cache` |
| `ImageModalityProcessor` | Image files / text-rendered images | `font_path`, `width`, `height`, `normalize_image`, `mean`, `std` |
| `FeaturesModalityProcessor` | `.npy` / `.pt` feature files | `skip_frames_stride`, `temporal_dimention_position`, `use_cache` |
| `SignwritingModalityProcessor` | FSW SignWriting strings | `custom_preprocessor_path`, `width`, `height`, `channels` |
| `TextModalityProcessor` | Text strings | `tokenizer`, `tokenizer_path`, `new_vocabulary`, `role` (`TextRole.INPUT` or `TextRole.TARGET`) |

`TextModalityProcessor` is the only processor that carries a tokenizer. The `role` parameter (a `TextRole` enum) controls batching behaviour:

| Role | Input | Output |
|---|---|---|
| `TextRole.INPUT` | List of strings | `(token_ids [B, L], attention_mask [B, L])` |
| `TextRole.TARGET` | List of dicts `{"target_prefix": …, "target": …}` | `(labels [B, L], None)` — padded with `-100` |

`new_vocabulary` is an optional path to a vocabulary file. When provided, the processor extends the tokenizer with the new tokens internally and exposes `self.new_tokens` (the added tokens) and `self.pretrained_tokenizer` (the unextended copy) as bridge attributes. These are used by setup scripts to derive tokenizer info for model construction without calling `load_tokenizers` separately — see TODO comments in `text_modality_processor.py` and the setup files.

---

## Layer 2: `ProcessorSlot`

A dataclass that binds a `ModalityProcessor` to dataset columns and output keys.

```python
@dataclass
class ProcessorSlot:
    processor: ModalityProcessor
    output_data_key: str           # key for the data tensor in the output batch
    output_mask_key: Optional[str] = None   # key for the mask tensor (None = no mask)
    column_map: Dict[str, str] = field(
        default_factory=lambda: {"signal": "signal"}
    )
    is_label: bool = False         # marks this slot as producing a loss target

    @property
    def primary_field(self) -> str:
        return next(iter(self.column_map))
```

### `column_map`

Maps **dataset item field names** (TSV column names) to **processor parameter names** (the keys inside the dict passed to `process_sample`).

- The **first key** is the *primary field*: its value is replaced with a preprocessed tensor by `_transform_get_items_output`.
- All **subsequent keys** are context-only: passed to `process_sample` but not written back into the dataset item (e.g. temporal bounds).

Default `{"signal": "signal"}` covers the standard single-field case.

**Multi-column (pose with temporal bounds):**
```python
column_map={
    "signal": "signal",               # primary field → written back as tensor
    "signal_start": "signal_start",   # context only
    "signal_end":   "signal_end",     # context only
}
```

**Label slot (needs two columns):**
```python
# TSV columns "decoder_prompt" and "output" map to processor params "target_prefix" and "target"
column_map={"decoder_prompt": "target_prefix", "output": "target"}
```

**Multi-input (two encoder streams sharing temporal bound column names):**
```python
# Pose slot
column_map={"pose_signal": "signal", "pose_signal_start": "signal_start", "pose_signal_end": "signal_end"}
# Video slot
column_map={"video_signal": "signal", "video_signal_start": "signal_start", "video_signal_end": "signal_end"}
```

### `is_label`

Marks a slot as producing a loss target. Does not affect processing logic — the `MultimodalMetaProcessor` iterates all slots identically. It is an annotation for callers (trainers, collators) to identify which output key carries the target sequence without hardcoding a key name.

---

## Layer 3: `MultimodalMetaProcessor`

Takes a flat list of `ProcessorSlot` objects and produces a complete `BatchFeature`.

```python
class MultimodalMetaProcessor(ProcessorMixin):
    def __init__(
        self,
        slots: List[ProcessorSlot],
        tokenizer=None,   # kept for HF ProcessorMixin compatibility
    ): ...
```

The meta-processor has **no knowledge of task structure**. It does not distinguish encoder inputs from labels from prompts. All semantic meaning lives in the processors and their slot configuration.

### `_transform_get_items_output` — dataset-level hook

Registered via `dataset.with_transform(processor._transform_get_items_output)`. Iterates over all slots and calls `slot.processor.process_sample()` on each item's primary field. Only writes the result back when `process_sample` returns a tensor — text slots (which are no-ops) do not corrupt their columns.

```python
dataset = dataset.with_transform(meta_processor._transform_get_items_output)
```

### `__call__` — collator-level call

Called by `DataCollatorMultimodalSeq2Seq` with a list of sample dicts. Iterates slots in declaration order. Skips a slot if its `output_data_key` is already present in the result.

### Save and load

Slots are serialised to `processor_config.json` — a flat list with each slot's processor class name, constructor kwargs, output keys, column map, and `is_label` flag. Reconstruction imports processor classes from `multimodalhugs.processors` by name.

```python
meta.save_pretrained("/path/to/save/")
loaded = MultimodalMetaProcessor.from_pretrained("/path/to/save/")
```

---

## Usage examples

**pose → text** (equivalent to legacy `Pose2TextTranslationProcessor`):

```python
MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=PoseModalityProcessor(reduce_holistic_poses=True),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
            column_map={"signal": "signal", "signal_start": "signal_start", "signal_end": "signal_end"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.TARGET),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.INPUT),
            output_data_key="encoder_prompt",
            output_mask_key="encoder_prompt_length_padding_mask",
            column_map={"encoder_prompt": "signal"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.INPUT),
            output_data_key="decoder_input_ids",
            output_mask_key="decoder_attention_mask",
            column_map={"decoder_prompt": "signal"},
        ),
    ],
    tokenizer=tokenizer,
)
```

**video + pose → text** (multi-encoder stream, no model changes required):

```python
MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=VideoModalityProcessor(),
            output_data_key="video_frames",
            output_mask_key="video_attention_mask",
            column_map={"video_signal": "signal", "video_signal_start": "signal_start", "video_signal_end": "signal_end"},
        ),
        ProcessorSlot(
            processor=PoseModalityProcessor(reduce_holistic_poses=True),
            output_data_key="pose_frames",
            output_mask_key="pose_attention_mask",
            column_map={"pose_signal": "signal", "pose_signal_start": "signal_start", "pose_signal_end": "signal_end"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.TARGET),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "target_prefix", "output": "target"},
        ),
    ],
    tokenizer=tokenizer,
)
```

**text + image → pose** (non-text output):

```python
MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.INPUT),
            output_data_key="input_ids",
            output_mask_key="attention_mask",
            column_map={"signal": "signal"},
        ),
        ProcessorSlot(
            processor=ImageModalityProcessor(),
            output_data_key="image_frames",
            output_mask_key="image_attention_mask",
            column_map={"image": "signal"},
        ),
        ProcessorSlot(
            processor=PoseModalityProcessor(),
            output_data_key="labels",
            is_label=True,
            column_map={"output_pose": "signal"},
        ),
    ],
    tokenizer=tokenizer,
)
```

---

## DataCollator simplification (step 5 — complete)

`DataCollatorMultimodalSeq2Seq` no longer constructs labels. When the processor is a `MultimodalMetaProcessor`, the collator delegates everything to it:

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

The tokenizer argument, label padding logic, and `create_seq2seq_labels_from_samples()` all remain available for the legacy processor path but are no longer used by the new design.

---

## Legacy backward compatibility (step 7 — complete)

The six original task processors are kept as thin `MultimodalMetaProcessor` subclasses in `multimodalhugs/processors/legacy/`. They construct the same flat `slots` list internally and expose the same named constructor parameters as before, so existing configs and code continue to work unchanged.

```python
class Pose2TextTranslationProcessor(MultimodalMetaProcessor):
    def __init__(self, tokenizer=None, reduce_holistic_poses=True, skip_frames_stride=None, **kwargs):
        if "slots" in kwargs:           # from_pretrained passthrough
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        super().__init__(slots=[...], tokenizer=tokenizer)
```

All six are re-exported from `multimodalhugs.processors` so existing import paths are unchanged.

---

---

## Step 8 — Declarative processor builder (complete)

`build_processor_from_config(processor_cfg)` in `multimodalhugs/training_setup/setup_utils.py` lets any YAML config opt in to constructing a `MultimodalMetaProcessor` directly, without touching Python code.

If `processor.slots` is absent the function returns `None` and every modality setup file falls through to its existing hardcoded construction — fully backward compatible.

### YAML format

```yaml
processor:
  slots:
    - processor_class: PoseModalityProcessor
      processor_kwargs:
        reduce_holistic_poses: true
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        signal: signal
        signal_start: signal_start
        signal_end: signal_end

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: /path/to/tokenizer  # each TextModalityProcessor loads its own tokenizer
        role: target
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix  # TSV column → processor param
        output: target

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: /path/to/tokenizer
        role: input
      output_data_key: encoder_prompt
      output_mask_key: encoder_prompt_length_padding_mask
      column_map:
        encoder_prompt: signal

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: /path/to/tokenizer
        role: input
      output_data_key: decoder_input_ids
      output_mask_key: decoder_attention_mask
      column_map:
        decoder_prompt: signal
```

Each slot dict mirrors the `ProcessorSlot` API:

| Key | Required | Description |
|---|---|---|
| `processor_class` | yes | Name of a `ModalityProcessor` subclass exported from `multimodalhugs.processors` |
| `output_data_key` | yes | Key for the data tensor in the output batch |
| `processor_kwargs` | no | Extra kwargs forwarded to the processor constructor |
| `output_mask_key` | no | Key for the optional mask tensor |
| `column_map` | no | Dataset column → processor param mapping (default `{"signal": "signal"}`) |
| `is_label` | no | Whether this slot produces the loss target (default `false`) |

Each `TextModalityProcessor` loads its own tokenizer from `tokenizer_path` in `processor_kwargs`. There is no global tokenizer injection — the `MultimodalMetaProcessor.tokenizer` is auto-derived from the first text slot after construction.

This path is used by all six modality setup files (`pose2text_training_setup.py`, `video2text_training_setup.py`, etc.). Each checks `build_processor_from_config` first; only if it returns `None` does it execute its hardcoded slot construction.

---

## Pending work

### Step 9 — Shorthand processor config format

The full declarative `slots:` format provides maximum flexibility but is verbose for the common single-modality case. A new shorthand layer on top of `build_processor_from_config` would let average users write a compact config while power users retain access to the full `slots:` syntax.

**Before (old flat format):**
```yaml
processor:
  text_tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__"
  custom_preprocessor_path: openai/clip-vit-base-patch32
  join_chw: false
  skip_frames_stride: 2
```

**After (current full slots format):**
```yaml
processor:
  slots:
    - processor_class: VideoModalityProcessor
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        join_chw: false
        skip_frames_stride: 2
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        signal: signal
        signal_start: signal_start
        signal_end: signal_end
    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__asl__"
        role: target
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix
        output: target
    # ... two more TextModalityProcessor slots
```

**Proposed shorthand:**
```yaml
processor:
  modality: video2text                           # expands to the standard 4-slot layout
  modality_kwargs:
    custom_preprocessor_path: openai/clip-vit-base-patch32
    join_chw: false
    skip_frames_stride: 2
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__"
```

`build_processor_from_config` would detect `modality:` (no `slots:` key) and expand it into the equivalent full slots list for the given modality. This gives three levels of interface:

| Level | Who uses it | Format |
|---|---|---|
| Shorthand (`modality:`) | Average user, standard tasks | 5–8 lines |
| Full slots (`slots:`) | Power user, custom pipelines | Full declaration |
| Python API | Developer, dynamic construction | `MultimodalMetaProcessor(slots=[...])` |

The hardcoded training setup files (`pose2text_training_setup.py`, etc.) would become the reference implementations for what each `modality:` shorthand expands to — and could eventually be removed once the shorthand covers all cases.

---

### Step 10 — Unified `multimodalhugs-setup` CLI

Currently `multimodalhugs-setup` dispatches on `--modality <name>` to one of six task-specific setup files. With the declarative config owning the full processor definition (steps 8 and 9), the modality dispatch becomes unnecessary.

**Goal:** A single general setup command that reads the config and builds all actors (dataset, processor, model) regardless of modality, without requiring `--modality`.

```bash
# Current — modality must be specified explicitly
multimodalhugs-setup --modality pose2text --config_path config.yaml --output_dir /out

# Target — modality inferred from config
multimodalhugs-setup --config_path config.yaml --output_dir /out
```

**Implementation sketch:**
- `multimodalhugs-setup` reads the config and infers the dataset type from `data.dataset_type` (or a new top-level `modality:` key)
- Calls `build_processor_from_config` to construct the processor (shorthand or full slots)
- Builds dataset and model using the same shared logic currently duplicated across the six setup files
- The six task-specific setup files become legacy/deprecated, kept only for backward compatibility

**Depends on:** Step 9 (shorthand format) should be in place so that setup via config is ergonomic for all users.

---

## Deferred work

### Step 11 — Dataset multi-column support

The TSV format and `GeneratorBasedBuilder` subclasses currently assume a fixed set of column names (`signal`, `signal_start`, `signal_end`, …). Multi-input scenarios like `video + pose → text` need additional columns with distinct names per stream:

```
pose_signal  pose_signal_start  pose_signal_end  video_signal  video_signal_start  video_signal_end  …
```

The `ProcessorSlot.column_map` mechanism is already designed to handle arbitrary column names. The remaining work is on the dataset side: making `GeneratorBasedBuilder` subclasses declare and yield additional columns.

### Step 12 — Multi-stream model support

`MultiModalEmbedderModel.forward()` currently accepts one encoder stream (`input_frames` / `attention_mask`). Three options for multi-stream support exist, in increasing complexity:

1. **Concatenate in the MetaProcessor** — merge all encoder slot outputs along the time axis before they reach the model. Zero model changes. Loses modality separation but validates the pipeline end-to-end immediately.
2. **Multiple feature extractors + merge** — `forward()` accepts `**encoder_inputs`, routes each to a separate `FeatureExtractor + MultimodalMapper` branch, then concatenates the resulting embeddings before the backbone encoder.
3. **Interleaved with separator tokens** — treat multi-modal input as a single sequence with special modality-boundary tokens.

Option 1 is the recommended first step when multi-stream experiments begin.

---

## Data flow

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
