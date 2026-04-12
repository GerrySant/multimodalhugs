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
```

If you need anything beyond these fields — different column names, extra slots, non-standard output keys — use the full `slots:` format instead.

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

## Reference: processor classes

This section lists every built-in `ModalityProcessor` subclass, its valid `processor_kwargs`, and the processor parameter names you can use in `column_map` values.

### `PoseModalityProcessor`

Loads `.pose` files and converts them to `[T, D]` tensors (T = frames, D = keypoint features).

**`processor_kwargs`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reduce_holistic_poses` | bool | `true` | Reduce full holistic pose to a smaller set of landmarks |
| `skip_frames_stride` | int | `null` | Keep every N-th frame (e.g. `2` = keep every other frame) |
| `signal_start_end_unit` | str | `"milliseconds"` | Unit for `signal_start` / `signal_end`: `"milliseconds"` or `"frames"`. `0`/`0` always loads the full file regardless of unit. |

**`column_map` processor param names:** `signal`, `signal_start`, `signal_end`

```yaml
column_map:
  signal: signal             # TSV column → processor param (required)
  signal_start: signal_start # clip start in milliseconds by default; see signal_start_end_unit
  signal_end: signal_end     # clip end in milliseconds by default; see signal_start_end_unit
```

---

### `VideoModalityProcessor`

Loads video files and converts them to `[T, C, H, W]` tensors (or `[T, C*H*W]` if `join_chw=true`).

**`processor_kwargs`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_preprocessor_path` | str | `null` | HF preprocessor identifier for frame resizing/normalisation (e.g. `openai/clip-vit-base-patch32`) |
| `skip_frames_stride` | int | `null` | Keep every N-th frame |
| `join_chw` | bool | `false` | Flatten channel, height, and width dimensions into one |
| `use_cache` | bool | `false` | Cache loaded videos in memory (speeds up repeated access) |
| `signal_start_end_unit` | str | `"milliseconds"` | Unit for `signal_start` / `signal_end`: `"milliseconds"` or `"frames"`. `0`/`0` always loads the full file regardless of unit. |

**`column_map` processor param names:** `signal`, `signal_start`, `signal_end`

```yaml
column_map:
  signal: signal
  signal_start: signal_start # clip start in milliseconds by default; see signal_start_end_unit
  signal_end: signal_end     # clip end in milliseconds by default; see signal_start_end_unit
```

---

### `ImageModalityProcessor`

Loads image files (PNG/JPEG/`.npy`) or renders text strings as images. Output is a `[C, H, W]` tensor.

**`processor_kwargs`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `font_path` | str | `null` | Path to a `.ttf` font file used when rendering text-as-image |
| `width` | int | `null` | Target image width in pixels |
| `height` | int | `null` | Target image height in pixels |
| `normalize_image` | bool | `true` | Normalise pixel values; requires `mean` and `std` when enabled |
| `mean` | list[float] | `null` | Per-channel mean for normalisation (e.g. `[0.5, 0.5, 0.5]`) |
| `std` | list[float] | `null` | Per-channel std for normalisation (e.g. `[0.5, 0.5, 0.5]`) |

> **Note:** `mean` and `std` are required when `normalize_image=true`.

**`column_map` processor param names:** `signal` only (no temporal bounds)

```yaml
column_map:
  signal: signal
```

---

### `FeaturesModalityProcessor`

Loads pre-computed feature files (`.npy`) and returns them as `[T, D]` tensors.

**`processor_kwargs`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skip_frames_stride` | int | `null` | Keep every N-th frame |
| `temporal_dimension_position` | int | `0` | Axis index of the temporal dimension in the loaded array |
| `use_cache` | bool | `true` | Cache loaded files in memory |

**`column_map` processor param names:** `signal` only (no temporal bounds; temporal slicing is not supported — load the full file)

```yaml
column_map:
  signal: signal
```

---

### `SignwritingModalityProcessor`

Converts ASCII SignWriting (FSW) strings to `[N, C, H, W]` symbol-image tensors.

**`processor_kwargs`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_preprocessor_path` | str | `null` | HF preprocessor identifier for image normalisation |
| `width` | int | `224` | Symbol image width in pixels |
| `height` | int | `224` | Symbol image height in pixels |
| `channels` | int | `3` | Number of image channels |
| `invert_frame` | bool | `true` | Invert pixel values (black symbols on white background → white on black) |

**`column_map` processor param names:** `signal` only

```yaml
column_map:
  signal: signal
```

---

### `TextModalityProcessor`

Tokenises text strings. Behaviour is controlled by the `role` parameter.

**`processor_kwargs`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tokenizer_path` | str | `null` | HF tokenizer identifier or local path (mutually exclusive with `tokenizer`) |
| `tokenizer` | object | `null` | Pre-loaded tokenizer instance (used in the Python API) |
| `new_vocabulary` | str | `null` | Comma-separated new tokens or path to a vocabulary file; extends the tokenizer internally |
| `role` | str | `input` | `input` — produce `(token_ids, attention_mask)`; `target` — produce `(labels, None)` with `-100` padding |

**`role` values explained:**

- **`role: input`** — used for encoder prompts and decoder context text. Returns `(token_ids [B, L], attention_mask [B, L])`. The `column_map` should map one TSV column to the `signal` processor param.
- **`role: target`** — used for the training target (labels). Concatenates a prefix and the target, appends EOS, and pads with `-100` (the standard ignore index for cross-entropy loss). The `column_map` must map two TSV columns to `target_prefix` and `target`.

**`column_map` processor param names depend on `role`:**

```yaml
# role: input — read one text column
column_map:
  encoder_prompt: signal   # or decoder_prompt: signal, or any single-column TSV field

# role: target — read two text columns and combine them
column_map:
  decoder_prompt: target_prefix   # optional prefix prepended before the target sequence
  output: target                  # the reference translation / generation target
```

---

## Reference: model-expected key names

The `output_data_key` and `output_mask_key` values you write in a slot become the keys in the batch dict that the model receives. **The correct key names depend on the model you are using** — they must exactly match the parameter names of the model's `forward()` method. A mismatch means the tensor is produced by the processor but silently ignored by the model, leading to wrong training results.

> **Rule:** look up the `forward()` signature of your model class and use its parameter names as `output_data_key` / `output_mask_key` values.

### `MultiModalEmbedderModel` (built-in)

The current built-in model (`type: multimodal_embedder`) expects:

```python
def forward(
    self,
    input_frames,                        # encoder visual/pose/feature input tensor
    attention_mask,                      # padding mask for input_frames
    encoder_prompt,                      # encoder text prompt token IDs
    encoder_prompt_length_padding_mask,  # padding mask for encoder_prompt
    decoder_input_ids,                   # decoder context token IDs
    decoder_attention_mask,              # padding mask for decoder_input_ids
    labels,                              # training target token IDs (with -100 padding)
    ...
)
```

| Slot purpose | `output_data_key` | `output_mask_key` |
|---|---|---|
| Main encoder input (pose, video, image, features, SignWriting) | `input_frames` | `attention_mask` |
| Encoder text prompt | `encoder_prompt` | `encoder_prompt_length_padding_mask` |
| Decoder context text | `decoder_input_ids` | `decoder_attention_mask` |
| Training labels | `labels` | *(omit — no mask needed)* |

### Custom or future models

If you register a new model with a different `forward()` signature, document its expected key names here (or in the model's own docs page) following the same table format. The processor config does not need to change — only the `output_data_key` and `output_mask_key` values in the slots need to match the new model's parameter names.

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
