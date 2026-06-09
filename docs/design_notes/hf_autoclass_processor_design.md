# Design Note: HF AutoClass Choice for MultimodalMetaProcessor

**Status:** Under consideration
**Related:** `multimodalhugs/processors/meta_processor.py`, `docs/processor_redesign.md`

---

## Motivation

### The coherence problem

Currently `MultimodalMetaProcessor` holds a `tokenizer` argument at the meta level:

```python
class MultimodalMetaProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, slots, tokenizer=None):
        self.slots = slots
        super().__init__(tokenizer=tokenizer)
```

But the tokenizer is not a meta-processor concern — it is a concern of `TextModalityProcessor`.
Every other `ModalityProcessor` subclass owns all its constructor arguments directly
(`PoseModalityProcessor` owns `reduce_holistic_poses`, `VideoModalityProcessor` owns
`custom_preprocessor_path`, etc.). `TextModalityProcessor` is the odd one out: it receives
its tokenizer from the outside because the meta-processor happens to need it too.

The coherent design is:

```yaml
processor:
  slots:
    - processor_class: TextModalityProcessor
      processor_kwargs:
        role: encoder
        tokenizer_path: /path/to/tokenizer   # owned by the slot, not the meta
      output_data_key: input_ids
```

`TextModalityProcessor.__init__` would accept either a pre-built `tokenizer` object or a
`tokenizer_path` string and load it internally.  `MultimodalMetaProcessor` would derive
`self.tokenizer` automatically by scanning slots — purely for `ProcessorMixin` compatibility,
not because it uses the tokenizer itself.

---

## Why we keep ProcessorMixin

### AutoProcessor is the right Auto class

The available HF Auto classes and their required base classes:

| Auto class | Base class / mixin | What it loads |
|---|---|---|
| `AutoProcessor` | `ProcessorMixin` | Multi-modal processors (text + vision + audio) |
| `AutoFeatureExtractor` | `FeatureExtractionMixin` | Single-modality feature extraction |
| `AutoImageProcessor` | `ImageProcessingMixin` | Image preprocessing only |
| `AutoTokenizer` | `PreTrainedTokenizer[Fast]` | Text tokenizers only |
| `AutoModel` | `PreTrainedModel` | Neural network models |
| `AutoConfig` | `PreTrainedConfig` | Configuration objects |

`AutoProcessor` is the correct choice: it is explicitly designed for processors that
orchestrate multiple modalities together. `AutoFeatureExtractor` and `AutoImageProcessor`
are single-modality and would not express the full scope of `MultimodalMetaProcessor`.

### What ProcessorMixin provides

| Feature | Notes |
|---|---|
| `save_pretrained` | Saves `preprocessor_config.json` + sub-processors |
| `from_pretrained` | Resolves local path / URL / HF Hub with caching |
| `push_to_hub` | Via `PushToHubMixin` |
| `register_for_auto_class()` | Enables `AutoProcessor.from_pretrained("org/model")` |
| `get_processor_dict()` | Loads `preprocessor_config.json` from local/Hub |

Dropping `ProcessorMixin` would mean losing `AutoProcessor` compatibility entirely —
`AutoProcessor` falls back to `AutoImageProcessor` / `AutoFeatureExtractor` only as a
last resort and those are not appropriate for this use case.

### The one problem: save_pretrained requires a non-None tokenizer

`ProcessorMixin.save_pretrained` iterates `self.attributes` and calls
`attribute.save_pretrained()` on each — it crashes if `self.tokenizer is None`.

This is fixable with a targeted override:

```python
def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
    os.makedirs(save_directory, exist_ok=True)
    config_path = os.path.join(save_directory, "preprocessor_config.json")
    with open(config_path, "w") as f:
        json.dump(self.to_dict(), f, indent=2)
    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(save_directory)
    if push_to_hub:
        self.push_to_hub(save_directory, **kwargs)
```

For all current tasks (everything-to-text) `self.tokenizer` is always non-None because at
least one text slot is always present. The None case only arises for future non-text-output
tasks (see deferred work below).

---

## Proposed changes

### 1. `TextModalityProcessor` — accept `tokenizer_path`

```python
def __init__(self, tokenizer=None, tokenizer_path=None, role="encoder"):
    if tokenizer is None and tokenizer_path is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    self.tokenizer = tokenizer
    self.role = role
```

### 2. `MultimodalMetaProcessor` — auto-derive tokenizer from slots

```python
def __init__(self, slots, tokenizer=None):
    self.slots = slots
    if tokenizer is None:
        tokenizer = next(
            (s.processor.tokenizer for s in slots
             if hasattr(s.processor, "tokenizer") and s.processor.tokenizer is not None),
            None,
        )
    super().__init__(tokenizer=tokenizer)
```

### 3. `MultimodalMetaProcessor` — override `save_pretrained`

As shown above: save config JSON directly, save tokenizer only when present.

### 4. `build_processor_from_config` — remove tokenizer injection

No more `inspect.signature` trick. `processor_kwargs` are forwarded as-is; each processor
class owns its own tokenizer loading.

### 5. Setup files — derive tokenizer from built processor

```python
proc = build_processor_from_config(processor_cfg)   # no tokenizer arg
if proc is None:
    # hardcoded path: still uses text_tokenizer_path + load_tokenizers
    ...
else:
    # TODO (future): for non-text-output tasks, tokenizer discovery needs revisiting
    tok = proc.tokenizer
    _, pre_tok, new = load_tokenizers(tok.name_or_path, new_vocabulary)
```

`text_tokenizer_path` at the `processor:` config level becomes unnecessary for the
declarative path (each text slot declares its own `tokenizer_path`).

---

## Deferred: non-text-output tasks

When a task has no `TextModalityProcessor` slot (e.g. text → pose), `proc.tokenizer`
will be `None`. Two issues arise:

1. The setup files need the tokenizer for model construction — they would need a separate
   `tokenizer_path` in the model config section instead.
2. `MultimodalMetaProcessor.save_pretrained` handles `None` gracefully (see above), but
   `from_pretrained` tries `AutoTokenizer.from_pretrained` in a try/except — already safe.

This is marked for revisit when the first non-text-output task is implemented.
