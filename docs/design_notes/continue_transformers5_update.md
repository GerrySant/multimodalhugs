# Handoff prompt — transformers 5.x compatibility update

Paste this file at the start of a new conversation to resume work on the
`update_transformers_compatibility` branch.

---

## About the user and this project

**User:** Gerard Sant — creator and primary maintainer of multimodalhugs. Expert in the HuggingFace
ecosystem (transformers, datasets, accelerate) and in sign language translation / multimodal NLP.
He is an ambitious collaborator who wants the framework to be truly HF-native, not just patched.

**multimodalhugs** is a modular HuggingFace extension for training, evaluating and deploying
multimodal AI models, primarily for sign language translation. It bridges non-text modalities
(pose sequences, video, images, SignWriting, precomputed features) with seq2seq text generation —
something HF does not support natively.

Core workflow: `mmhugs-setup` → build dataset/processor/model from YAML → `mmhugs-train` →
`mmhugs-generate`.

Three-component model (`MultiModalEmbedderModel`):
1. **FeatureExtractor** — wraps pretrained vision models (CLIP, ViT)
2. **MultimodalMapper** — maps feat_dim → d_model (linear / adapter / cnn_adapter)
3. **Backbone** — HF seq2seq model (M2M-100, mBART) loaded via AutoModelForSeq2SeqLM

---

## How the user likes to work — follow these rules exactly

### Environment
- Always use the conda env `mmhugs`, not `.venv`.
- Run commands with `conda run -n mmhugs <command>`.

### Commit style
- Every commit title must start with a tag: `[fix]`, `[feat]`, `[refactor]`, `[test]`, `[docs]`, `[arch]`
- Example: `[fix] Remove tie_encoder_decoder attribute removed in transformers 5.x`
- Body must be **very descriptive**: what changed, which files, and why.
- **Never add a `Co-Authored-By:` line.**
- Commit after **each resolved incompatibility** — one commit per fix, not batched.

### Tests
- **Before modifying, adding, or removing any test**: explain what needs to change and why, then
  request explicit user approval. Do NOT touch tests without approval.
- After approval: always update `tests/TESTS.md` to reflect the additions/modifications/removals.

### Transformers compatibility work — most important rule
**Always adapt multimodalhugs to the new HF API — never patch around it.** Concrete examples:
- When an attribute is removed from model configs, remove it from multimodalhugs configs too — do
  not add `getattr(..., fallback)` shims.
- When a return type changes, add a **general** extraction helper that works for any model, not a
  one-off hardcode.
- When a class/method is removed, implement the replacement using the new API, not a `try/except`
  that catches the `AttributeError`.
- Before writing any fix, look at how HuggingFace implements the same thing in the nearest
  analogous model (e.g. `M2M100ForConditionalGeneration`) and follow that pattern.
- The goal is genuine API alignment, not "not crashing".

### Documentation
- Every incompatibility found must be documented in `docs/transformers_compatibility.md` before
  committing the fix. Format: heading with the incompatibility name, files affected, description
  of the change, and the fix applied.

---

## Current branch and state

**Branch:** `update_transformers_compatibility`
**Goal:** Migrate from `transformers<=4.44.2` to `transformers==5.5.0`

Run the test suite with:
```bash
conda run -n mmhugs python -m pytest tests/ --ignore=tests/e2e_overfitting -q
```

**Current state: 11 failing tests, 398 passing.**

---

## What has already been done (committed)

| Commit | Incompatibility fixed |
|---|---|
| `84c171e` | `MODEL_WITH_LM_HEAD_MAPPING_NAMES` removed → `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` |
| `d7b72c3` | `send_example_telemetry` removed from `transformers.utils` |
| `fba35ed` | `ProcessorMixin.__init__` validates typed modality components — overrode `get_attributes()` on `MultimodalMetaProcessor`; `tokenizer` redesigned as read-only `@property` |
| `146c78b` | `add_special_tokens()` dropped `replace_additional_special_tokens` kwarg; `additional_special_tokens` → `extra_special_tokens`; `AutoTokenizer` raises `ValueError` instead of `OSError` on missing files |
| `65e71fb` | `tie_encoder_decoder` removed from all seq2seq model configs |
| `21fb9be` | `max_length` moved from model configs to `GenerationConfig` — removed from `MultiModalEmbedderConfig` entirely; `model.max_length` → `model.generation_config.max_length` in training/generate scripts |
| `4c8a4fa` | Six fixes: `_tied_weights_keys` list→dict; `_no_split_modules`/`_keep_in_fp32_modules` list→set; `get_image_features()` returns `ModelOutput` (general extraction helper); `GenerationMixin` must be declared explicitly; `__getattr__` delegation on `MultiModalEmbedderConfig`; `_reorder_cache` removed from backbone — updated to `cache.reorder_cache()` |

Additionally (committed after the main session — check `git log` to confirm):
- `tests/test_model_only/configs/test_model_only.yaml` — updated from legacy processor format to
  slot-based format
- `tests/test_model_only/test_model_only.py` — fixture updated to use `build_processor_from_config`;
  `test_backbone_shared_weights_are_tied` added; `test_model_maxlength_is_correct` removed
- `tests/test_config/test_multimodal_embedder_config.py` — all 4 max_length tests removed (the
  attribute no longer exists)
- `tests/TESTS.md` — kept in sync throughout

---

## Remaining tasks (in order)

### Task 1 — Fix 11 failing tests (needs approval before touching tests)

**Root cause:** Tests construct `MultimodalMetaProcessor(slots=[...], tokenizer=tokenizer)`.
After the processor redesign (PR #70), `MultimodalMetaProcessor.__init__` no longer accepts a
`tokenizer` kwarg — the tokenizer is derived automatically from the first text slot via a
`@property`. The `tokenizer=` argument needs to be removed from all constructor calls in the tests.

**Files to check:**
- `tests/test_data/test_datacollator.py` (6 failing tests in `TestDataCollatorWithMetaProcessor`)
- `tests/test_data/test_processor_regression.py` (5 failing tests in `TestMetaProcessorXxxGolden`)

Per workflow rules: **explain the change and ask for explicit approval before editing any test.**
After approval, update `tests/TESTS.md`.

### Task 2 — Fix `multilingual_seq2seq_trainer.py` for transformers 5.x

Full analysis is in `docs/design_notes/multilingual_seq2seq_trainer.md`. Three changes:

**a) Remove the `gen_kwargs` override (bug fix)**

Line 130 in `prediction_step`:
```python
gen_kwargs = self.model.generation_config.to_dict()  # ← remove this line
```
This unconditionally populates `gen_kwargs`, so the subsequent check for `_gen_kwargs` (which
carries training args like `--generation_max_length`) is always skipped. The 5.x base class
handles generation config correctly without this line.

**b) Rename `tokenizer` → `processing_class`**

In transformers 5.x, `Seq2SeqTrainer.__init__` and `Trainer.__init__` renamed the `tokenizer`
parameter to `processing_class`. `Trainer` now sets `self.processing_class`, not `self.tokenizer`.

Changes in `multilingual_seq2seq_trainer.py`:
- `__init__` signature: `tokenizer: Optional[...] = None` → `processing_class: Optional[...] = None`
- `super().__init__()` call: `tokenizer=tokenizer` → `processing_class=processing_class`
- `visualize_generation()`: all `self.tokenizer` references → `self.processing_class`

**c) Update `synced_gpus` check to match 5.x**

```python
# Current (4.44.2 pattern — line 146):
default_synced_gpus = True if is_deepspeed_zero3_enabled() else False

# 5.x pattern:
from transformers.integrations import is_fsdp_managed_module
default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
```

Document each sub-fix in `docs/transformers_compatibility.md` and commit each separately.

### Task 3 — Update `pyproject.toml` dependency pin

Change `transformers<=4.44.2` to the appropriate new range (e.g. `transformers>=5.5.0,<6.0`).
Verify `torch` version constraint is still appropriate.

### Task 4 — Pre-merge cleanup (before opening/merging PR #81)

Delete these three one-off verification scripts that must not be merged:
- `tests/assets/verify_meta_processor.py`
- `tests/assets/verify_setup_and_train_loading.py`
- `tests/assets/sample_data.py`

Also revert the `conftest.py` changes that delegated fixture bodies to `sample_data.py` — restore
the inline fixture definitions, since `sample_data.py` will no longer exist.

---

## Key file locations

```
multimodalhugs/multilingual_seq2seq_trainer.py        # trainer — main pending work (Task 2)
multimodalhugs/processors/meta_processor.py           # MultimodalMetaProcessor
multimodalhugs/models/multimodal_embedder/
  modeling_multimodal_embedder.py                     # MultiModalEmbedderModel
  configuration_multimodal_embedder.py                # MultiModalEmbedderConfig
docs/transformers_compatibility.md                    # document every incompatibility here
docs/design_notes/multilingual_seq2seq_trainer.md     # trainer analysis and pending 5.x updates
tests/TESTS.md                                        # update when touching tests
tests/test_data/test_datacollator.py                  # 6 failing tests (Task 1)
tests/test_data/test_processor_regression.py          # 5 failing tests (Task 1)
pyproject.toml                                        # dependency pins to update (Task 3)
```

---

## Persistent memory

All memory files live in:
```
/Users/sant/.claude/projects/-Users-sant-repositories-multimodalhugs/memory/
```

Read `MEMORY.md` there first for the full index. The most important files are:
- `feedback_transformers_update_workflow.md` — workflow rules for this branch
- `feedback_commit_format.md` — commit style
- `feedback_environment.md` — use conda env `mmhugs`
- `feedback_tests_md.md` — always update TESTS.md
- `project_verify_scripts_cleanup.md` — pre-merge cleanup details
