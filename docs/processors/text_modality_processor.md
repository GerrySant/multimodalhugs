# TextModalityProcessor

`TextModalityProcessor` tokenises text strings for use in any model pipeline that
consumes text — encoder-decoder, encoder-only, decoder-only, or any future pipeline
type.  It has two roles, selected by the `role` parameter, which control how the
tokenised output is formatted.

No optional dependency is required beyond `transformers`.

---

## Roles

### `TextRole.INPUT` — any text input

`process_batch` receives a list of plain text strings and tokenises them.  Returns
standard padded token ids and an attention mask.

```
data:  token_ids       [B, L]  int64, padded to the longest sequence in the batch
mask:  attention_mask  [B, L]  int64  (1 = real token, 0 = padding)
```

Use this for any slot that feeds text into a model component — a source language tag,
a task instruction, a decoder conditioning token, a classification prompt, or any
other text input regardless of the model architecture.

### `TextRole.TARGET` — supervision labels

`process_batch` receives a list of dicts, each with keys `"target_prefix"` and
`"target"`, and builds a label sequence for supervised training:

```
label = tokenize(target_prefix) + tokenize(target) + [EOS]
```

Labels are padded with `−100`, the standard `CrossEntropyLoss` ignore index in
HuggingFace Transformers.  Returns:

```
data:  labels  [B, L]  int64, padded with −100
mask:  None              (no attention mask — loss functions handle −100 directly)
```

If any sample has `target=None`, returns `(None, None)`.

`target_prefix` is an optional conditioning token prepended before the target
(e.g. a language tag or task prefix).  Leave `decoder_prompt` empty in the TSV when
no prefix is needed — an empty string produces zero prefix tokens.

---

## Column map

Each slot has a `column_map` that maps TSV column names to the parameter names the
processor expects.  This allows any TSV column to be used as text input without
renaming the dataset.

**`TextRole.INPUT`** — single column:

```yaml
column_map:
  encoder_prompt: signal   # TSV column "encoder_prompt" → processor param "signal"
```

**`TextRole.TARGET`** — two columns, renamed to `target_prefix` and `target`:

```yaml
column_map:
  decoder_prompt: target_prefix   # TSV column → processor param
  output: target
```

---

## Vocabulary extension (`new_vocabulary`)

`new_vocabulary` accepts a path to a plain-text file (one token per line) or a
comma-separated string of tokens.  These are added as special tokens to the tokenizer.

After extension:
- `self.tokenizer` — the extended tokenizer used for all encoding.
- `self.new_tokens` — list of added tokens.
- `self.pretrained_tokenizer` — the original unextended tokenizer.

Example — adding custom modality tags:

```yaml
processor_kwargs:
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__,__fsl__"
```

All slots in a pipeline that share the same `tokenizer_path` + `new_vocabulary` pair
share the same extended tokenizer instance via an internal cache in
`MultimodalMetaProcessor.from_pretrained`.

---

## YAML config examples

The following examples show the three standard text slots used in a seq2seq pipeline.
For other pipeline types, configure the slots to match the model's expected inputs.

### Source-side text input (e.g. encoder prompt or source language tag)

```yaml
- processor_class: TextModalityProcessor
  processor_kwargs:
    tokenizer_path: facebook/m2m100_418M
    new_vocabulary: "__asl__"
    role: input
  output_data_key: encoder_prompt
  output_mask_key: encoder_prompt_length_padding_mask
  column_map:
    encoder_prompt: signal
```

### Any other text input (e.g. decoder conditioning, classification prompt)

```yaml
- processor_class: TextModalityProcessor
  processor_kwargs:
    tokenizer_path: facebook/m2m100_418M
    new_vocabulary: "__asl__"
    role: input
  output_data_key: decoder_prompt_ids
  output_mask_key: decoder_prompt_mask
  column_map:
    decoder_prompt: signal
```

### Supervision labels

```yaml
- processor_class: TextModalityProcessor
  processor_kwargs:
    tokenizer_path: facebook/m2m100_418M
    new_vocabulary: "__asl__"
    role: target
  output_data_key: labels
  is_label: true
  column_map:
    decoder_prompt: target_prefix   # optional prefix before the target text
    output: target
```

All three slots are generated automatically when using the `pipeline:` shorthand.

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tokenizer` | tokenizer instance \| `None` | `None` | Pre-built HuggingFace tokenizer. When provided, `tokenizer_path` is ignored for loading but still stored for serialisation. |
| `tokenizer_path` | `str \| None` | `None` | HuggingFace model ID or local path. Used to load the tokenizer when `tokenizer` is `None`. |
| `new_vocabulary` | `str \| None` | `None` | Vocabulary file path or comma-separated tokens to add as special tokens to the tokenizer. |
| `role` | `TextRole \| str` | `TextRole.INPUT` | `"input"` — tokenise strings, return ids + attention mask. `"target"` — build label sequences from `target_prefix` + `target` + EOS, padded with −100. |

---

## Common mistakes

**Wrong `column_map` for the labels slot**

`TextRole.TARGET` expects dict keys `"target_prefix"` and `"target"`.  If the TSV
column names differ, the `column_map` must rename them:

```yaml
column_map:
  decoder_prompt: target_prefix   # TSV name : processor param name
  output: target
```

Forgetting this mapping causes a `KeyError` inside `_process_label_batch`.

**Using `role: target` for a non-label slot**

`TextRole.TARGET` pads sequences with `−100`.  Using it for any slot other than the
loss target means the model receives `−100` tokens as input, which is meaningless for
all tokenizers.  Any slot that feeds text into a model component must use `role: input`.

**Mismatched `tokenizer_path` across slots**

All text slots in a pipeline should use the same `tokenizer_path` and
`new_vocabulary`.  If they differ, each slot loads and extends the tokenizer
independently, which may produce inconsistent vocabulary sizes across slots.
