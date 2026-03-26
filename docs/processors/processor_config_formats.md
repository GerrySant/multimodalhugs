# Processor Config Formats

There are three ways to configure a `MultimodalMetaProcessor` in a YAML config file. They are processed by `build_processor_from_config()` in `multimodalhugs/training_setup/setup_utils.py` and all produce the same type of object — a `MultimodalMetaProcessor` with a flat list of `ProcessorSlot` objects.

| Format | Key | Best for |
|---|---|---|
| [Shorthand](#1-shorthand-pipeline) | `pipeline:` | Standard single-modality tasks |
| [Full slots](#2-full-slots-slots) | `slots:` | Custom pipelines, non-standard layouts |
| [Python API](#3-python-api) | — | Dynamic construction in setup scripts |

All three formats are supported simultaneously. You can write the full `slots:` format even if a `pipeline:` shorthand exists for your task — there is no penalty.

---

## 1. Shorthand (`pipeline:`)

The shorthand is the most concise option. You declare the pipeline type and shared options; the framework generates the standard 4-slot layout automatically.

**Supported `pipeline` values:** `pose2text`, `video2text`, `image2text`, `features2text`, `signwriting2text`, `text2text`.

**Minimal example:**
```yaml
processor:
  pipeline: pose2text
  tokenizer_path: facebook/m2m100_418M
```

**Full example with all optional fields:**
```yaml
processor:
  pipeline: video2text
  tokenizer_path: facebook/m2m100_418M   # required — shared by all text slots
  new_vocabulary: "__asl__"              # optional — token(s) to add to the tokenizer
  modality_kwargs:                       # optional — forwarded to the modality slot's constructor
    custom_preprocessor_path: openai/clip-vit-base-patch32
    join_chw: false
    skip_frames_stride: 2
  slot_overrides:                        # optional — sparse per-slot patches
    encoder_prompt:
      column_map:
        my_column: signal                # replace the default column name for this slot only
```

### What the shorthand expands to

Every `pipeline:` shorthand expands to exactly 4 slots:

| # | `output_data_key` | Processor | Role | What it reads |
|---|---|---|---|---|
| 1 | `input_frames` + `attention_mask` | Modality processor (varies by pipeline) | — | `signal` (+ `signal_start`/`signal_end` for temporal modalities) |
| 2 | `labels` | `TextModalityProcessor` | `target` | `decoder_prompt` + `output` TSV columns |
| 3 | `encoder_prompt` + `encoder_prompt_length_padding_mask` | `TextModalityProcessor` | `input` | `encoder_prompt` TSV column |
| 4 | `decoder_input_ids` + `decoder_attention_mask` | `TextModalityProcessor` | `input` | `decoder_prompt` TSV column |

Slot 1 is the only one that varies between pipelines:

| `pipeline` value | Slot 1 processor class |
|---|---|
| `pose2text` | `PoseModalityProcessor` |
| `video2text` | `VideoModalityProcessor` |
| `image2text` | `ImageModalityProcessor` |
| `features2text` | `FeaturesModalityProcessor` |
| `signwriting2text` | `SignwritingModalityProcessor` |
| `text2text` | `TextModalityProcessor` (role=input) |

### `slot_overrides`

`slot_overrides` lets you patch specific slots without writing the full `slots:` list. The key is the `output_data_key` of the slot to modify. Dict fields (`column_map`, `processor_kwargs`) are shallow-merged; scalar fields (`output_mask_key`, `is_label`) are replaced.

```yaml
processor:
  pipeline: pose2text
  tokenizer_path: facebook/m2m100_418M
  slot_overrides:
    input_frames:
      processor_kwargs:
        reduce_holistic_poses: true   # added to the PoseModalityProcessor constructor
    encoder_prompt:
      column_map:
        lang_tag: signal              # read from 'lang_tag' column instead of 'encoder_prompt'
```

---

## 2. Full slots (`slots:`)

The full slots format gives complete control over every slot. Use it when the shorthand does not cover your setup (e.g. multiple encoder streams, non-standard output keys, or a novel modality).

Each entry in the `slots:` list becomes one `ProcessorSlot`. The slots are processed in declaration order.

```yaml
processor:
  slots:
    - processor_class: PoseModalityProcessor    # any ModalityProcessor subclass
      processor_kwargs:                          # forwarded directly to the constructor
        reduce_holistic_poses: true
        skip_frames_stride: 2
      output_data_key: input_frames             # key for the tensor in the model batch
      output_mask_key: attention_mask           # key for the padding mask (omit if not needed)
      column_map:                               # dataset column → processor param mapping
        signal: signal
        signal_start: signal_start
        signal_end: signal_end

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__asl__"
        role: target                            # TextRole.TARGET: produces labels
      output_data_key: labels
      is_label: true                            # marks this slot as the loss target
      column_map:
        decoder_prompt: target_prefix           # TSV column → processor param (rename absorbed here)
        output: target

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__asl__"
        role: input                             # TextRole.INPUT: produces (ids, mask)
      output_data_key: encoder_prompt
      output_mask_key: encoder_prompt_length_padding_mask
      column_map:
        encoder_prompt: signal

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__asl__"
        role: input
      output_data_key: decoder_input_ids
      output_mask_key: decoder_attention_mask
      column_map:
        decoder_prompt: signal
```

The above is exactly what `pipeline: pose2text` with `new_vocabulary: "__asl__"` expands to.

### Per-slot fields

| Field | Required | Default | Description |
|---|---|---|---|
| `processor_class` | yes | — | Name of a `ModalityProcessor` subclass exported from `multimodalhugs.processors` |
| `output_data_key` | yes | — | Key written for the data tensor in the batch dict |
| `output_mask_key` | no | `null` | Key written for the padding mask tensor; omit when no mask is needed |
| `column_map` | no | `{signal: signal}` | Mapping from dataset TSV column names to processor parameter names |
| `is_label` | no | `false` | Marks this slot as the loss target (consumed by trainers and collators) |
| `processor_kwargs` | no | `{}` | Extra keyword arguments forwarded to the processor class constructor |

### Multi-encoder-stream example

The full slots format makes multi-encoder inputs straightforward — just add more modality slots:

```yaml
processor:
  slots:
    - processor_class: VideoModalityProcessor
      output_data_key: video_frames
      output_mask_key: video_attention_mask
      column_map:
        video_signal: signal
        video_signal_start: signal_start
        video_signal_end: signal_end

    - processor_class: PoseModalityProcessor
      output_data_key: pose_frames
      output_mask_key: pose_attention_mask
      column_map:
        pose_signal: signal
        pose_signal_start: signal_start
        pose_signal_end: signal_end

    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        role: target
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix
        output: target

    # ... additional text slots as needed
```

Note: multi-encoder-stream model support is tracked in issue #72.

---

## 3. Python API

For setup scripts that need to construct a processor programmatically (e.g. when parameters are computed at runtime), use the Python API directly:

```python
from multimodalhugs.processors import PoseModalityProcessor, TextModalityProcessor
from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextRole

processor = MultimodalMetaProcessor(slots=[
    ProcessorSlot(
        processor=PoseModalityProcessor(reduce_holistic_poses=True),
        output_data_key="input_frames",
        output_mask_key="attention_mask",
        column_map={"signal": "signal", "signal_start": "signal_start", "signal_end": "signal_end"},
    ),
    ProcessorSlot(
        processor=TextModalityProcessor(
            tokenizer_path="facebook/m2m100_418M",
            new_vocabulary="__asl__",
            role=TextRole.TARGET,
        ),
        output_data_key="labels",
        is_label=True,
        column_map={"decoder_prompt": "target_prefix", "output": "target"},
    ),
    ProcessorSlot(
        processor=TextModalityProcessor(
            tokenizer_path="facebook/m2m100_418M",
            new_vocabulary="__asl__",
            role=TextRole.INPUT,
        ),
        output_data_key="encoder_prompt",
        output_mask_key="encoder_prompt_length_padding_mask",
        column_map={"encoder_prompt": "signal"},
    ),
    ProcessorSlot(
        processor=TextModalityProcessor(
            tokenizer_path="facebook/m2m100_418M",
            new_vocabulary="__asl__",
            role=TextRole.INPUT,
        ),
        output_data_key="decoder_input_ids",
        output_mask_key="decoder_attention_mask",
        column_map={"decoder_prompt": "signal"},
    ),
])
```

The Python API is used internally by the legacy task-specific setup files (`pose2text_training_setup.py`, etc.) and by `build_processor_from_config` after expanding the YAML config.

---

## Equivalence

The three formats are semantically equivalent for standard cases. This pose-to-text config:

```yaml
processor:
  pipeline: pose2text
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__"
  modality_kwargs:
    reduce_holistic_poses: true
```

…produces the same `MultimodalMetaProcessor` as the full `slots:` form above, which produces the same result as calling the Python API directly with the same arguments.

---

## Choosing a format

- **Use `pipeline:`** when you are training a standard single-modality task and the default 4-slot layout covers your needs. It is the most concise and least error-prone option.
- **Use `slots:`** when you need non-standard output keys, additional encoder streams, a non-text label modality, or any configuration that falls outside the 4-slot template.
- **Use the Python API** in setup scripts when processor parameters are computed at runtime (e.g. derived from the dataset), or when you need access to bridge attributes like `processor.new_tokens` or `processor.pretrained_tokenizer` before saving.
