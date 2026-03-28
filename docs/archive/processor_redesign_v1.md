# Processor Redesign: Modality-First Architecture (v1 — Historical)

> **This document is a historical design proposal.** It describes the architecture as originally conceived before implementation. The final design differs in several ways (flat `slots` list instead of named constructor params, `TextRole` enum instead of role strings, `target_prefix`/`target` param names, etc.). For the current architecture, see [`processor_redesign.md`](processor_redesign.md) and [`processors/processors_overview.md`](processors/processors_overview.md).

---

## Motivation

The current processor design (`Pose2TextTranslationProcessor`, `Video2TextTranslationProcessor`, etc.) couples three separate concerns into a single class:

1. **What data to load** — modality-specific file loading and feature extraction
2. **How to combine inputs** — orchestrating encoder inputs, prompts, and labels
3. **What output format to produce** — always text, always tokenized in the DataCollator

This makes it impossible to:
- Reuse modality logic across tasks (e.g., `PoseProcessor` shared by `pose2text` and `pose2pose`)
- Handle multiple encoder inputs (`video + pose → text`)
- Handle non-text outputs (`text + image → pose`)

---

## Proposed Architecture

Three layers replace the current task-specific processors:

```
ModalityProcessor        — knows one modality (pose, video, text, image, ...)
ProcessorSlot            — binds a ModalityProcessor to a TSV column + a forward() key name
MultimodalMetaProcessor  — orchestrates all slots; replaces task-specific processors
```

---

## Layer 1: `ModalityProcessor`

A pure modality-specific class with no knowledge of the task or the surrounding pipeline. Each modality gets one class.

```python
class ModalityProcessor:
    """
    Handles a single modality end-to-end: loading, preprocessing, padding, masking.
    Has no knowledge of task structure (what is encoder input vs. label, etc.).
    """

    def process_sample(self, raw_value, **kwargs) -> torch.Tensor:
        """
        Loads and preprocesses a single raw value (e.g., a file path or string).
        Called at dataset.with_transform() time — no padding, no batching.
        """
        raise NotImplementedError

    def process_batch(self, samples: List[torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Takes a list of pre-loaded tensors (from process_sample), pads them to
        the same length, and returns (data_tensor, mask_tensor).
        Called inside the collator after the full batch is assembled.
        """
        raise NotImplementedError
```

**Concrete implementations:**

| Class | Modality | Notes |
|---|---|---|
| `PoseModalityProcessor` | Pose sequences | Wraps current `_pose_file_to_tensor` logic |
| `VideoModalityProcessor` | Video frames | Wraps current video loading logic |
| `ImageModalityProcessor` | Images | Wraps current image loading logic |
| `TextModalityProcessor` | Text (tokenized) | Holds a tokenizer; handles labels, prompts |
| `FeaturesModalityProcessor` | Precomputed features | Loads `.npy`/`.pt` feature files |
| `SignWritingModalityProcessor` | SignWriting notation | Wraps current SignWriting logic |

`TextModalityProcessor` is the one that carries a tokenizer. It is configurable to handle different text roles: a raw prompt, a label sequence (`decoder_prompt + output + EOS`), or plain text.

---

## Layer 2: `ProcessorSlot` — the routing solution

A `ProcessorSlot` explicitly declares:
- **where** to read the raw data from (TSV column name)
- **which** `ModalityProcessor` to use
- **where** to write the processed tensors (forward() argument names)

```python
@dataclass
class ProcessorSlot:
    source_column: str             # TSV column to read from (e.g., "signal", "output")
    processor: ModalityProcessor
    output_data_key: str           # forward() argument name for the data tensor
    output_mask_key: Optional[str] = None  # forward() argument name for the mask tensor
```

This is the answer to the routing question: **the mapping from modality to forward() argument is declared explicitly per slot**, not inferred. There is no magic.

### Examples

```python
# Pose input → forward() receives "input_frames" + "attention_mask"
ProcessorSlot(
    source_column="signal",
    processor=PoseModalityProcessor(reduce_holistic_poses=True),
    output_data_key="input_frames",
    output_mask_key="attention_mask",
)

# Video input → forward() receives "video_frames" + "video_attention_mask"
ProcessorSlot(
    source_column="video_signal",
    processor=VideoModalityProcessor(skip_frames_stride=2),
    output_data_key="video_frames",
    output_mask_key="video_attention_mask",
)

# Text label → forward() receives "labels" (no mask needed)
ProcessorSlot(
    source_column="output",
    processor=TextModalityProcessor(tokenizer, role="label"),
    output_data_key="labels",
)
```

---

## Layer 3: `MultimodalMetaProcessor`

Replaces all task-specific processors. Accepts a list of encoder slots plus dedicated slots for labels and prompts.

```python
class MultimodalMetaProcessor(ProcessorMixin):
    def __init__(
        self,
        encoder_slots: List[ProcessorSlot],
        label_slot: ProcessorSlot,
        encoder_prompt_slot: Optional[ProcessorSlot] = None,
        decoder_prompt_slot: Optional[ProcessorSlot] = None,
        tokenizer=None,   # kept for HF ProcessorMixin compatibility
    ): ...

    def _transform_get_items_output(self, batch):
        """
        Delegates to each slot's process_sample().
        Registered with dataset.with_transform() — runs at DataLoader time, before batching.
        """
        for slot in self._all_slots():
            batch[slot.source_column] = [
                slot.processor.process_sample(v) for v in batch[slot.source_column]
            ]
        return batch

    def __call__(self, batch: List[Dict]) -> BatchFeature:
        """
        Runs after the full batch is assembled (inside the DataCollator).
        Calls process_batch() on each slot and merges the results.
        """
        result = {}
        for slot in self.encoder_slots:
            data, mask = slot.processor.process_batch([s[slot.source_column] for s in batch])
            result[slot.output_data_key] = data
            if slot.output_mask_key and mask is not None:
                result[slot.output_mask_key] = mask

        if self.encoder_prompt_slot:
            data, mask = self.encoder_prompt_slot.processor.process_batch(...)
            result["encoder_prompt"] = data
            result["encoder_prompt_length_padding_mask"] = mask

        if self.decoder_prompt_slot:
            data, mask = self.decoder_prompt_slot.processor.process_batch(...)
            result["decoder_input_ids"] = data
            result["decoder_attention_mask"] = mask

        label_data, _ = self.label_slot.processor.process_batch(
            [s[self.label_slot.source_column] for s in batch]
        )
        result[self.label_slot.output_data_key] = label_data

        return BatchFeature(result)
```

### Usage examples

**Simple: pose → text** (equivalent to current `Pose2TextTranslationProcessor`)
```python
MetaProcessor(
    encoder_slots=[
        ProcessorSlot("signal", PoseModalityProcessor(reduce_holistic_poses=True),
                      output_data_key="input_frames", output_mask_key="attention_mask"),
    ],
    label_slot=ProcessorSlot("output", TextModalityProcessor(tokenizer, role="label"),
                             output_data_key="labels"),
    encoder_prompt_slot=ProcessorSlot("encoder_prompt", TextModalityProcessor(tokenizer, role="prompt"),
                                      output_data_key="encoder_prompt", output_mask_key="encoder_prompt_length_padding_mask"),
)
```

**Multi-input: video + pose → text**
```python
MetaProcessor(
    encoder_slots=[
        ProcessorSlot("video_signal", VideoModalityProcessor(),
                      output_data_key="video_frames", output_mask_key="video_attention_mask"),
        ProcessorSlot("pose_signal",  PoseModalityProcessor(),
                      output_data_key="pose_frames",  output_mask_key="pose_attention_mask"),
    ],
    label_slot=ProcessorSlot("output", TextModalityProcessor(tokenizer, role="label"),
                             output_data_key="labels"),
)
```

**Non-text output: text + image → pose**
```python
MetaProcessor(
    encoder_slots=[
        ProcessorSlot("text_input", TextModalityProcessor(tokenizer, role="encoder"),
                      output_data_key="encoder_input_ids", output_mask_key="encoder_attention_mask"),
        ProcessorSlot("image",      ImageModalityProcessor(),
                      output_data_key="image_frames",      output_mask_key="image_attention_mask"),
    ],
    label_slot=ProcessorSlot("output", PoseModalityProcessor(),
                             output_data_key="labels"),
)
```

---

## Fixing the text-output assumption

Currently, `DataCollatorMultimodalSeq2Seq` hard-codes label creation via `create_seq2seq_labels_from_samples()`, which tokenizes `decoder_prompt + output + EOS`. This is the source of the text-output assumption.

**In the new design, label processing moves into the MetaProcessor's `label_slot`.**

- `TextModalityProcessor(tokenizer, role="label")` handles the `decoder_prompt + output + EOS` concatenation and tokenization
- A future `PoseModalityProcessor` as a label slot would load pose files instead
- The DataCollator is reduced to:
  1. Call `processor(batch)` — all modality processing, including labels, happens here
  2. If the model has `prepare_decoder_input_ids_from_labels()`, call it on the result

The DataCollator no longer needs a tokenizer argument — that knowledge lives inside the `TextModalityProcessor` in the `label_slot`.

### `TextModalityProcessor` roles

The `role` parameter controls how `TextModalityProcessor` handles its input:

| Role | Input columns used | Output |
|---|---|---|
| `"label"` | `decoder_prompt` + `output` (concatenated) | `{"labels": tokenized_ids}` |
| `"prompt"` | the configured column | `{"data": token_ids, "mask": attention_mask}` |
| `"encoder"` | the configured column | `{"data": token_ids, "mask": attention_mask}` |

This avoids needing separate processor classes for each text use case.

---

## DataCollator simplification

```python
@dataclass
class DataCollatorMultimodalSeq2Seq:
    processor: MultimodalMetaProcessor
    model: Optional[Any] = None

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

The tokenizer argument, label padding logic, and `create_seq2seq_labels_from_samples()` all move into `TextModalityProcessor`.

---

## HF compatibility: save/load

`ProcessorMixin` uses a static `attributes` list (e.g., `["tokenizer", "frame_preprocessor"]`) to serialize sub-processors. A dynamic slot composition breaks this assumption.

**Solution:** Override `save_pretrained()` and `from_pretrained()` in `MultimodalMetaProcessor`:

- `save_pretrained(path)`:
  - Save each slot's processor to `path/<slot_name>/`
  - Save a `meta_processor_config.json` describing the slot composition (processor class names, source columns, output keys, roles)
- `from_pretrained(path)`:
  - Read `meta_processor_config.json`
  - Reconstruct each `ModalityProcessor` from its subdirectory
  - Rebuild the `ProcessorSlot` list and the `MultimodalMetaProcessor`

This gives full flexibility while keeping the HF `from_pretrained()` interface that users expect.

---

## What requires model and dataset changes later

The processor design above is forward-compatible with multi-encoder models. The changes needed elsewhere when implementing multi-input scenarios:

### Dataset
TSV format needs additional columns for each encoder input. For `video + pose → text`:
```
signal  video_signal  signal_start  signal_end  encoder_prompt  decoder_prompt  output
path/to/pose.pose  path/to/video.mp4  0  5000  ...  ...  gloss
```
Each `GeneratorBasedBuilder` subclass would need to declare and yield these extra columns.

### Model
`MultiModalEmbedderModel.forward()` currently accepts one encoder stream (`input_frames` / `attention_mask`). For multi-stream input, three options exist (in increasing complexity):

1. **Concatenate in the MetaProcessor** — merge all encoder slot outputs along the time axis before they reach the model. Zero model changes. Loses modality separation but works immediately.
2. **Multiple feature extractors + merge** — forward() accepts `**encoder_inputs`, routes each key to a separate `FeatureExtractor` + `MultimodalMapper` branch, then concatenates the resulting embeddings before the backbone encoder. Medium effort.
3. **Interleaved with separator tokens** — treat multi-modal input as a single sequence with special modality-boundary tokens. Closest to how large multimodal models work. Requires rethinking the backbone interface.

Option 1 (concatenation in the MetaProcessor) allows the new processor design to be validated with the current model before committing to model changes.

---

## Migration path

1. Implement `ModalityProcessor` base + all concrete modality processors
2. Implement `ProcessorSlot` dataclass
3. Implement `MultimodalMetaProcessor` with HF save/load override
4. Move label processing out of `DataCollatorMultimodalSeq2Seq` into `TextModalityProcessor`
5. Simplify `DataCollatorMultimodalSeq2Seq`
6. Update `training_setup/` per-modality setup scripts to build `MetaProcessor` from slots instead of task-specific processors
7. Deprecate (or wrap) old task-specific processor classes for backward compatibility
8. Update dataset TSV handling and `GeneratorBasedBuilder` subclasses for multi-column inputs (when needed)
9. Extend `MultiModalEmbedderModel.forward()` for multi-stream input (when needed)
