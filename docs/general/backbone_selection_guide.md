# Pretrained backbone selection guide for seq2seq pipelines

This guide explains how to choose a seq2seq backbone for multimodalhugs and how to
structure your dataset TSV columns and YAML config accordingly.

The critical question is: **where does your backbone expect task and language
conditioning — on the encoder side or on the decoder side?** The answer determines
how you fill the `encoder_prompt` and `decoder_prompt` TSV columns and what token
IDs to use.

---

## Quick reference

| Backbone family | Representative models | Language/task conditioning | decoder_prompt column |
|---|---|---|---|
| Encoder-conditioned | T5, mT5, ByT5 | encoder input (prefix) | **empty** |
| Decoder-conditioned A | M2M100, mBART-50-many-to-many | language token at decoder position 1 | language token |
| Decoder-conditioned B | mBART-cc25 | language token IS decoder_start_token_id | **empty** |
| Monolingual | BART, MarianMT | no language token needed | **empty** |

---

## How multimodalhugs uses encoder_prompt and decoder_prompt

Every multimodalhugs dataset sample has two optional text fields:

| TSV column | Maps to | What it does |
|---|---|---|
| `encoder_prompt` | `encoder_prompt` tensor prepended to encoder input | Provides source context, modality tag, or task instruction |
| `decoder_prompt` | `decoder_prompt_ids` tensor used to seed generation | Provides target language token or decoder prefix for generation |

During training, `decoder_prompt` is concatenated with the target `output` and an EOS
token to form the full label sequence:

```
labels = tokenize(decoder_prompt) + tokenize(output) + [EOS]
```

After shifting right, `decoder_input_ids` becomes:

```
decoder_input_ids = [decoder_start_token_id] + tokenize(decoder_prompt) + tokenize(output)
```

During generation, the decoder is seeded with:

```
generation prefix = [decoder_start_token_id] + tokenize(decoder_prompt)
```

This means: **`decoder_prompt` and `decoder_start_token_id` together fully determine what
the decoder sees before it starts generating.** Getting this right is the key to train/
inference alignment.

---

## How to find decoder_start_token_id for any model

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("your/model")
print(cfg.decoder_start_token_id)   # the token id at decoder position 0 during training
```

---

## Family 1 — Encoder-conditioned: T5, mT5, ByT5

### How they work

All task and language conditioning goes in the **encoder input**. The decoder is
intentionally neutral: it always starts from `<pad>` (token id 0, the `pad_token_id`),
for every language, every task.

```
Training:
  encoder: "translate sign language to English: <source frames>"
  labels:  [Hello, world, </s>]             (no language token)

  decoder_start_token_id = 0  (<pad>)
  decoder_input_ids = [<pad>, Hello, world]  ← <pad> always at position 0

Generation:
  decode starts: [<pad>]                     ← automatic
  generates:  Hello  world  </s>
```

### Dataset TSV structure

```tsv
signal	signal_start	signal_end	encoder_prompt	decoder_prompt	output
/data/video_001.mp4	0	4500	translate sign language to English:		Hello world
/data/video_002.mp4	0	3200	translate sign language to English:		Good morning
```

- `encoder_prompt`: the task/language instruction (e.g. `"translate sign language to English:"`)
- `decoder_prompt`: **leave empty** — T5 does not use decoder-side conditioning

### Multilingual setup (mT5)

Different target languages are specified in `encoder_prompt`, not `decoder_prompt`:

```tsv
signal	signal_start	signal_end	encoder_prompt	decoder_prompt	output
/data/video_001.mp4	0	4500	translate sign language to English:		Hello world
/data/video_002.mp4	0	3200	translate sign language to French:		Bonjour monde
/data/video_003.mp4	0	5100	translate sign language to German:		Guten Morgen
```

### YAML config example

```yaml
data:
  dataset_type: video2text
  train_metadata_path: /data/train.tsv
  dev_metadata_path:   /data/dev.tsv
  test_metadata_path:  /data/test.tsv

processor:
  pipeline: video2text
  tokenizer_path: google/mt5-small

model:
  type: multimodal_embedder
  backbone_type: t5
  pretrained_backbone: google/mt5-small
  feat_dim: 512
  multimodal_mapper_type: linear
  # decoder_start_token_id is 0 (<pad>) for all T5 variants — no need to set it
```

### Checking language token IDs (not needed for T5)

T5 does not use language tokens in the decoder. If you need language-specific
tokenization, use mT5 and put the language tag in `encoder_prompt`.

---

## Family 2A — Decoder-conditioned: M2M100, mBART-50-many-to-many

### How they work

Language conditioning lives in the decoder at **position 1**. Position 0 is always the
`decoder_start_token_id` (`</s>`, token id 2), which is the same for every language:

```
Training:
  encoder: <source frames>
  labels:  [__en__, Hello, world, </s>]     (language token is first label)

  decoder_start_token_id = 2  (</s>)
  decoder_input_ids = [</s>, __en__, Hello, world]
                        ↑       ↑
                  always    language token — varies per sample

Generation:
  generation prefix = [</s>, __en__]        ← built from decoder_start_token_id + decoder_prompt
  generates: Hello  world  </s>
```

This means different samples in the same batch can target different languages — the
language token at position 1 carries per-sample conditioning.

### Dataset TSV structure

```tsv
signal	signal_start	signal_end	encoder_prompt	decoder_prompt	output
/data/asl_001.mp4	0	4500	__asl__	__en__	Hello world
/data/asl_002.mp4	0	3200	__asl__	__en__	Good morning
/data/fsl_001.mp4	0	5100	__fsl__	__fr__	Bonjour monde
```

- `encoder_prompt`: source modality tag (e.g. `__asl__` for American Sign Language)
- `decoder_prompt`: target language token (e.g. `__en__`, `__fr__`, `__de__`)
- `output`: the target text **without** the language token (it is already in `decoder_prompt`)

### Finding language token IDs

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

# M2M100 language token format: __xx__
print(tok.get_lang_id("en"))    # e.g. 250004
print(tok.get_lang_id("fr"))    # e.g. 250008
print(tok.convert_ids_to_tokens([250004, 250008]))
```

### YAML config example

```yaml
data:
  dataset_type: video2text
  train_metadata_path: /data/train.tsv
  dev_metadata_path:   /data/dev.tsv
  test_metadata_path:  /data/test.tsv

processor:
  pipeline: video2text
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__,__fsl__"   # add any custom vocabulary tokens

model:
  type: multimodal_embedder
  backbone_type: m2m_100
  pretrained_backbone: facebook/m2m100_418M
  feat_dim: 512
  multimodal_mapper_type: linear
  # decoder_start_token_id is 2 (</s>) — set automatically from the pretrained backbone
```

### mBART-50-many-to-many

Identical pattern to M2M100. Only the token format differs:

```tsv
signal	encoder_prompt	decoder_prompt	output
/data/video_001.mp4	en_XX	fr_XX	Bonjour monde
```

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# mBART-50 language token format: xx_XX
print(tok.lang_code_to_id["en_XX"])   # e.g. 250008
print(tok.lang_code_to_id["fr_XX"])   # e.g. 250023
```

```yaml
model:
  backbone_type: mbart
  pretrained_backbone: facebook/mbart-large-50-many-to-many-mmt
```

---

## Family 2B — Decoder-conditioned: mBART-cc25 (older style)

> ⚠ **Not fully supported in the current version of multimodalhugs.**
> See the note at the end of this section. If you need a decoder-conditioned backbone,
> use M2M100 or mBART-50 (Family 2A) instead.

### How it works in the original HuggingFace setup

mBART-cc25's `shift_tokens_right` is unique: it does **not** use `decoder_start_token_id`
from the config. Instead, it reads the **last non-pad token** of the label sequence and
moves it to position 0. The HuggingFace mBART-cc25 tokenizer exploits this by appending
the target language token at the end of every label sequence:

```
HF tokenizer output for a fr target:
  labels = [tok1, tok2, tok3, fr_FR_id]   (language token at the END, no EOS)

shift_tokens_right(labels, pad_token_id):
  takes last non-pad token = fr_FR_id
  decoder_input_ids = [fr_FR_id, tok1, tok2, tok3]   ← correct
```

During generation, `model.generate()` is called with `decoder_start_token_id = fr_FR_id`
set in the model config (or via `forced_bos_token_id`), so position 0 matches training.

### Why this does not work with the current multimodalhugs processors

`TextModalityProcessor(role=target)` builds labels as:

```
labels = tokenize(target_prefix) + tokenize(output) + [eos_token_id]
```

For mBART-cc25 with `decoder_prompt = ""` (empty) and `output = "Hello world"`:

```
labels = [tok1, tok2, </s>]

shift_tokens_right(labels, pad_token_id):
  last non-pad token = </s> = 2
  decoder_input_ids = [2, tok1, tok2]   ← </s> at position 0, not the language token
```

Meanwhile, `model.generate()` starts from `decoder_start_token_id` in the config (e.g.
`fr_FR_id`). **Train position 0 ≠ generation position 0 → mismatch.** This was confirmed
by a programmatic test against a randomly-initialized mBART model.

### What is needed to support mBART-cc25

The fix requires a new `TextRole.TARGET_SUFFIX` in `TextModalityProcessor`:

```
TARGET_SUFFIX: labels = tokenize(output) + tokenize(target_suffix)
               # target_suffix = language token, NO eos_token_id appended
```

With `target_suffix = "fr_FR"`:
```
labels = [tok1, tok2, fr_FR_id]
shift_tokens_right → [fr_FR_id, tok1, tok2]   ← matches generation ✓
```

This role would be used in the labels slot's `column_map` with
`decoder_prompt → target_suffix` (instead of the current `decoder_prompt → target_prefix`).
See the design notes for the implementation plan.

### Recommended alternative

Use M2M100 or mBART-50 (Family 2A) instead of mBART-cc25. They use a standard
`shift_tokens_right(labels, pad_id, decoder_start_token_id)` that does not require
language tokens in the label suffix, support per-sample language tokens in
`decoder_prompt`, and are strictly more capable models.

### Limitation (when supported)

mBART-cc25 fixes one target language per model instance. To support multiple target
languages, either use M2M100 / mBART-50 (Family 2A) or train separate models per
language direction.

---

## Family 3 — Monolingual: BART, MarianMT

### How they work

No language conditioning is needed. The decoder always starts from `</s>` and generates
text in the single language the model was trained on.

```
Training:
  decoder_start_token_id = 2  (</s>)
  decoder_input_ids = [</s>, tok1, tok2, ...]

Generation:
  model.generate() starts with [</s>]
  → no decoder_prompt needed
```

### Dataset TSV structure

```tsv
signal	signal_start	signal_end	encoder_prompt	decoder_prompt	output
/data/video_001.mp4	0	4500		 	Hello world
```

Both `encoder_prompt` and `decoder_prompt` can be empty for purely monolingual tasks.

### YAML config example

```yaml
model:
  backbone_type: bart
  pretrained_backbone: facebook/bart-large
```

---

## Decoder sequence comparison across families

This table shows what the decoder sees at each position during training for a
sample with output `"Hello world"` targeting English, across all families.

```
M2M100 / mBART-50:
  decoder_input_ids: [</s>,  __en__,  Hello,  world]
  labels:            [__en__,  Hello,  world,  </s>]
  position 0 = </s>   (always, regardless of language)
  position 1 = __en__ (language token, varies per sample)

mBART-cc25 (en→fr model) — NOT YET SUPPORTED, shown for reference:
  HF native labels:   [Hello,  world,  fr_FR]   (lang token at END, no EOS)
  decoder_input_ids:  [fr_FR,  Hello,  world]   (last label token lifted to pos 0)
  position 0 = fr_FR (shift_tokens_right uses last label token, not decoder_start_token_id)

  Current multimodalhugs (broken — labels end with EOS, not lang token):
  labels:            [Hello,  world,  </s>]
  decoder_input_ids: [</s>,   Hello,  world]   ← position 0 = </s>, NOT fr_FR

T5 / mT5 / ByT5:
  decoder_input_ids: [<pad>,  Hello,  world]
  labels:            [Hello,  world,  </s>]
  position 0 = <pad> (always, task/language in encoder input)

BART:
  decoder_input_ids: [</s>,   Hello,  world]
  labels:            [Hello,  world,  </s>]
  position 0 = </s>  (always, no language conditioning)
```

---

## Choosing the right backbone for your task

| Use case | Recommended backbone | Reason |
|---|---|---|
| Sign language → fixed target language | M2M100 or mBART-50 | Multilingual, per-sample language tokens, well-pretrained on parallel data |
| Sign language → multiple target languages in same dataset | M2M100 or mBART-50 | Per-sample `decoder_prompt` supports different languages within one training run |
| Instruction-following or multitask | mT5 or ByT5 | Task prefix goes in `encoder_prompt`; clean encoder-side conditioning |
| Monolingual summarization or paraphrase | BART | No language conditioning overhead |
| Byte-level processing (scripts without word segmentation) | ByT5 | Character/byte-level tokenisation |

---

## Common mistakes

**Using mBART-cc25 with the current version of multimodalhugs**

mBART-cc25 is not currently supported. Its `shift_tokens_right` expects the target
language token at the **end** of the label sequence (not a prefix, and no EOS token), but
`TextModalityProcessor(role=target)` appends EOS instead. This causes a train/inference
mismatch: the decoder sees `</s>` at position 0 during training but `fr_FR_id` during
generation. Use M2M100 or mBART-50 (Family 2A) instead.

**Putting the task prefix in `decoder_prompt` for T5**

T5 always starts from `<pad>`. A task prefix in `decoder_prompt` would become position 1
of the decoder, which is unusual and may degrade performance. For T5, put the task
prefix in `encoder_prompt`.

**Using an empty `decoder_prompt` with M2M100**

Without a language token in `decoder_prompt`, the generation prefix is just `[</s>]` and
the model does not know which language to generate. Either set `decoder_prompt` to the
correct language token in every TSV row, or set `forced_bos_token_id` in the generation
config if all samples target the same language.

**Mismatching encoder_prompt and decoder_prompt language tags**

For M2M100, `encoder_prompt` typically carries the *source* language tag and
`decoder_prompt` the *target* language tag. Swapping them or using the same tag for
both will confuse the model.
