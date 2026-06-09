# TextModalityProcessor

`TextModalityProcessor` tokenises text strings for two different roles in a seq2seq
pipeline: encoder/decoder input prompts (`TextRole.INPUT`) and target label sequences
(`TextRole.TARGET`).

No optional dependency is required beyond `transformers`.

---

## Roles

`TextModalityProcessor` behaviour is determined by the `role` parameter:

### `TextRole.INPUT` — encoder or decoder prompt

`process_batch` receives a list of plain text strings and returns:

```
data:  token_ids       [B, L]  int64, padded
mask:  attention_mask  [B, L]  int64
```

Use this for the `encoder_prompt` slot and the `decoder_prompt_ids` slot.

### `TextRole.TARGET` — labels for seq2seq loss

`process_batch` receives a list of dicts, each with keys `"target_prefix"` and
`"target"`, and returns:

```
data:  labels  [B, L]  int64, padded with −100 (CrossEntropyLoss ignore index)
mask:  None
```

The label sequence for each sample is:
```
tokenize(target_prefix) + tokenize(target) + [EOS]
```

`target_prefix` is typically the `decoder_prompt` column (e.g. `"__en__"` for M2M100);
`target` is the `output` column.  If any sample has `target=None`, returns
`(None, None)`.

---

## Column map for the labels slot

The labels slot reads **two** TSV columns and renames them to the keys
`"target_prefix"` and `"target"` that `TextRole.TARGET` expects:

```yaml
- processor_class: TextModalityProcessor
  processor_kwargs:
    tokenizer_path: facebook/m2m100_418M
    role: target
  output_data_key: labels
  is_label: true
  column_map:
    decoder_prompt: target_prefix   # TSV column → processor param name
    output: target
```

---

## Vocabulary extension (`new_vocabulary`)

`new_vocabulary` accepts a path to a plain-text file (one token per line) or a
comma-separated string of tokens.  Tokens are added as special tokens to the tokenizer
via `add_special_tokens`.

After extension:
- `self.tokenizer` — the extended tokenizer (used for all encoding/decoding).
- `self.new_tokens` — list of added tokens.
- `self.pretrained_tokenizer` — the original unextended tokenizer.

This is the mechanism used to add custom vocabulary such as sign language modality
tags (`__asl__`, `__fsl__`):

```yaml
processor_kwargs:
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__,__fsl__"
```

All text slots in a pipeline that share the same `tokenizer_path` + `new_vocabulary`
pair share the same extended tokenizer instance (via an internal cache in
`MultimodalMetaProcessor.from_pretrained`).

---

## YAML config examples

### Encoder prompt slot

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

### Decoder prompt slot

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

### Labels slot

```yaml
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
```

All three slots are generated automatically when using the `pipeline:` shorthand.

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tokenizer` | tokenizer instance \| `None` | `None` | Pre-built HuggingFace tokenizer. When provided, `tokenizer_path` is ignored for loading but still stored for serialisation. |
| `tokenizer_path` | `str \| None` | `None` | HuggingFace model ID or local path. Used to load the tokenizer when `tokenizer` is `None`. |
| `new_vocabulary` | `str \| None` | `None` | Path to a vocabulary file or comma-separated tokens to add as special tokens. |
| `role` | `TextRole \| str` | `TextRole.INPUT` | `"input"` — tokenise strings, return ids + mask. `"target"` — build labels from `target_prefix` + `target` + EOS, pad with −100. |

---

## Common mistakes

**Wrong `column_map` for the labels slot**

`TextRole.TARGET` expects dict keys `"target_prefix"` and `"target"`.  If the TSV
column names are different (e.g. `"decoder_prompt"` and `"output"`), the `column_map`
must rename them:

```yaml
column_map:
  decoder_prompt: target_prefix   # ← TSV column name: processor param name
  output: target
```

Forgetting this mapping causes a `KeyError` inside `_process_label_batch`.

**Using `role: target` for a prompt slot**

Prompt slots must use `role: input`.  Using `role: target` for `encoder_prompt` or
`decoder_prompt_ids` causes them to be treated as label sequences (padded with −100),
which will break the model forward pass.

**Mismatched `tokenizer_path` across slots**

All text slots should use the same `tokenizer_path` and `new_vocabulary`.  If they
differ, each slot loads and extends the tokenizer independently, which may produce
inconsistent vocabulary sizes.
