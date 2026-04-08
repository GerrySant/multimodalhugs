# Transformers Compatibility Notes

Tracking breaking changes found when updating multimodalhugs from `transformers<=4.44.2` to `transformers==5.5.0`.

---

## Breaking Changes Found

### 1. `MODEL_WITH_LM_HEAD_MAPPING_NAMES` removed
**Files affected:** `multimodalhugs/models/utils.py`, `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`, `multimodalhugs/models/multimodal_embedder/configuration_multimodal_embedder.py`

**Change:** `MODEL_WITH_LM_HEAD_MAPPING_NAMES` was removed from `transformers.models.auto.modeling_auto`. The equivalent for seq2seq models is `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES` (already used), and for causal LM models use `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`.

**Fix:** Replaced `MODEL_WITH_LM_HEAD_MAPPING_NAMES` with `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` in `utils.py`. Removed unused import from `modeling_multimodal_embedder.py` and `configuration_multimodal_embedder.py`.

---

### 2. `send_example_telemetry` removed from `transformers.utils`
**Files affected:** `multimodalhugs/tasks/translation/translation_training.py`, `multimodalhugs/tasks/translation/translation_generate.py`

**Change:** `send_example_telemetry` was removed from `transformers.utils` in 5.x.

**Fix:** Removed the import and all call sites. This was optional HF telemetry with no functional impact.

---

### 3. `ProcessorMixin.__init__` now validates typed modality components
**Files affected:** `multimodalhugs/processors/meta_processor.py`

**Change:** In transformers 5.x, `ProcessorMixin.get_attributes()` introspects `__init__` signatures for parameter names matching `MODALITY_TO_AUTOPROCESSOR_MAPPING` (i.e. `tokenizer`, `image_processor`, `feature_extractor`, `video_processor`, `audio_processor`). `ProcessorMixin.__init__` then requires properly typed non-None instances for each discovered component. In 4.x, `attributes = []` was sufficient to suppress this behavior.

**Root cause:** `MultimodalMetaProcessor` has no fixed typed modality components at the meta level — all components live in `ProcessorSlot` objects. However, legacy subclasses (e.g. `Text2TextTranslationProcessor`) have `tokenizer` in their `__init__` signature as a build-time convenience parameter, which `get_attributes()` incorrectly interprets as a declared component.

**Fix:** Override `get_attributes()` on `MultimodalMetaProcessor` to return `[]`. This is architecturally correct: the meta-processor declares no modality components — its components are managed by slots. The override is inherited by all subclasses, protecting the entire family.

**Design note:** The `tokenizer` attribute on `MultimodalMetaProcessor` was also redesigned as a read-only `@property` that derives the tokenizer from the first text slot, rather than storing it as a constructor parameter. This properly reflects that tokenizer ownership belongs to `TextModalityProcessor`, not the meta-processor.

---

### 4. `add_special_tokens()` no longer accepts `replace_additional_special_tokens` kwarg
**Files affected:** `multimodalhugs/utils/tokenizer_utils.py`

**Change:** `PreTrainedTokenizerBase.add_special_tokens()` in transformers 5.x removed the `replace_additional_special_tokens` keyword argument.

**Fix:** Removed the kwarg. Default behavior in 5.x is to append (not replace), which matches the intended usage.

---

### 5. Fast tokenizer instantiation requires `sentencepiece` or `tiktoken`
**Files affected:** Tests using `AutoTokenizer.from_pretrained` on the tiny test tokenizer

**Change:** In transformers 5.x, loading a fast tokenizer backed by SentencePiece requires `sentencepiece` to be explicitly installed.

**Fix:** `sentencepiece` added as an explicit dependency and installed in the dev environment.

---

### 6. `AutoTokenizer.from_pretrained` raises `ValueError` (not `OSError`) when no tokenizer files exist
**Files affected:** `multimodalhugs/processors/meta_processor.py`

**Change:** In transformers 4.x, calling `AutoTokenizer.from_pretrained` on a directory without tokenizer files raises `OSError`. In transformers 5.x, the same situation raises `ValueError`.

**Fix:** Changed `except OSError:` to `except (OSError, ValueError):` in `MultimodalMetaProcessor.from_pretrained` so that non-text pipelines (processors saved without a tokenizer) continue to work across both versions.

---

### 7. `tie_encoder_decoder` removed from all seq2seq model configs
**Files affected:** `multimodalhugs/models/multimodal_embedder/configuration_multimodal_embedder.py`

**Change:** `tie_encoder_decoder` was a boolean attribute on `M2M100Config`, `MarianConfig`, and other seq2seq configs in transformers 4.x. It was removed entirely in 5.x — it is no longer a supported attribute on any `PretrainedConfig` subclass.

**Fix:** Removed the line `self.tie_encoder_decoder = self.backbone_config.tie_encoder_decoder` from `MultiModalEmbedderConfig.__init__`. Used `getattr(self.backbone_config, "tie_word_embeddings", False)` for the remaining `tie_word_embeddings` read.

---

### 8. `max_length` removed from model configs — moved to `GenerationConfig`
**Files affected:** `multimodalhugs/models/multimodal_embedder/configuration_multimodal_embedder.py`, `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`, `multimodalhugs/tasks/translation/translation_training.py`, `multimodalhugs/tasks/translation/translation_generate.py`

**Change:** In transformers 5.x, `max_length` was removed from all model configs (`M2M100Config`, `MarianConfig`, etc.) and moved to `GenerationConfig`. Accessing `backbone_config.max_length` raises `AttributeError` in 5.x.

**Fix:** Removed `max_length` and `use_backbone_max_length` from `MultiModalEmbedderConfig` entirely — following the same design decision HuggingFace applied to all their model configs. Removed `self.max_length = config.max_length` from `MultiModalEmbedderModel`. Generation length is now managed via `model.generation_config.max_length` (the standard HF 5.x pattern). Updated `translation_training.py` and `translation_generate.py` to use `model.generation_config.max_length` instead of `model.max_length`.

---

### 9. `additional_special_tokens` renamed to `extra_special_tokens`
**Files affected:** `multimodalhugs/utils/tokenizer_utils.py`, `tests/test_model_only/test_model_only.py`

**Change:** In transformers 5.x, `PreTrainedTokenizerFast` is now `TokenizersBackend`. The `.additional_special_tokens` attribute no longer exists; it is replaced by `.extra_special_tokens`. The same rename applies to dict keys passed to `add_special_tokens()` and constructor kwargs (though those retain backward-compat shims that auto-convert the old name).

**Fix:**
- `tokenizer_utils.py` line 44: `additional_special_tokens=` kwarg → `extra_special_tokens=`
- `tokenizer_utils.py` line 97: `{'additional_special_tokens': ...}` dict key → `{'extra_special_tokens': ...}`
- `test_model_only.py` line 42: `src_tokenizer.additional_special_tokens` → `src_tokenizer.extra_special_tokens`
