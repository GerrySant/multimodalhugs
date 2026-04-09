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

---

### 10. `_tied_weights_keys` changed from list to dict; `find_tied_parameters` override removed
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In transformers 5.x, `_tied_weights_keys` on `PreTrainedModel` subclasses changed from a `list` of tied parameter names to a `dict` mapping each tied parameter to its source (e.g. `{"lm_head.weight": "model.shared.weight"}`). The old code used `find_tied_parameters()` (from `accelerate`) to auto-detect tied weights and store the result as a list, then assigned it to `backbone._tied_weights_keys`. This overrode the backbone's correct 5.x dict with a list, causing `get_expanded_tied_weights_keys()` to fail with `AttributeError: 'list' object has no attribute 'keys'`.

**Fix:** Removed the `find_tied_parameters` usage and the `backbone._tied_weights_keys` override entirely from both `build_model` and `_init_backbone`. The backbone model's built-in `_tied_weights_keys` dict is correct and sufficient. Added an explicit `backbone.tie_weights()` call (guarded with `hasattr`) after `load_state_dict` in `build_model` to ensure tied weight references are restored after weight copying.

---

### 11. `_no_split_modules` and `_keep_in_fp32_modules` changed from list to set
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In transformers 5.x, `_no_split_modules` and `_keep_in_fp32_modules` on `PreTrainedModel` subclasses are sets, not lists. The code concatenated these with `+` (list concatenation), causing `TypeError: can only concatenate list (not "set") to list`.

**Fix:** Wrapped all `getattr(module, "_no_split_modules", [])` and `getattr(module, "_keep_in_fp32_modules", [])` reads with `list()` before concatenation.

---

### 12. `get_image_features()` returns `ModelOutput` instead of tensor
**Files affected:** `multimodalhugs/modules/feature_extractor.py`

**Change:** In transformers 5.x, `CLIPModel.get_image_features()` returns `BaseModelOutputWithPooling` instead of a plain tensor.

**Fix:** After calling `get_image_features()`, check if the result is already a tensor. If not, extract the primary feature tensor in a model-agnostic way: prefer `pooler_output` (projected/pooled representation) when available, fall back to `last_hidden_state` for models without a pooling head.

---

### 13. `GenerationMixin` no longer inherited through `PreTrainedModel`
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In transformers 5.x, `PreTrainedModel` no longer inherits from `GenerationMixin`. All generative models must explicitly declare `GenerationMixin` in their inheritance. This matches the pattern used by all transformers 5.x generative models (e.g. `M2M100ForConditionalGeneration(M2M100PreTrainedModel, GenerationMixin)`).

**Fix:** Changed `class MultiModalEmbedderModel(PreTrainedModel)` to `class MultiModalEmbedderModel(PreTrainedModel, GenerationMixin)`.

---

### 14. `MultiModalEmbedderConfig` missing generation-machinery attributes; `__getattr__` delegation added
**Files affected:** `multimodalhugs/models/multimodal_embedder/configuration_multimodal_embedder.py`

**Change:** `GenerationMixin` and `DynamicCache` in transformers 5.x access generation-relevant attributes (e.g. `num_hidden_layers`, `vocab_size`) directly on `self.config`. `MultiModalEmbedderConfig` is a composite config that wraps a backbone config; these attributes live on `backbone_config`, not on the outer config.

**Fix:** Added `__getattr__` to `MultiModalEmbedderConfig` that delegates unknown attribute lookups to `backbone_config`. Standard Python `__getattr__` is only called when the attribute is not found through normal lookup, so own attributes always take precedence.

---

### 16. `torch<2.6` constraint blocks model loading in transformers 5.x
**Files affected:** `pyproject.toml`

**Change:** transformers 5.x added `check_torch_load_is_safe()` which hard-blocks `torch.load` when torch < 2.6, raising `ValueError: Due to a serious vulnerability issue in torch.load, we now require torch >= v2.6`. This affects loading any model whose weights are stored in `.bin` format (not safetensors). The original `torch<2.6` cap was added for transformers 4.x where torch 2.6's stricter `weights_only` default caused HF deprecation warnings; in 5.x the constraint has the opposite effect.

**Fix:** Changed `torch<2.6` to `torch>=2.6` in `pyproject.toml`. In CI, also updated the Python version from 3.8 to 3.11, as transformers 5.x requires Python ≥3.9.

---

### 17. `tokenizer.convert_tokens_to_ids(None)` no longer returns `None`
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In transformers 4.x, calling `tokenizer.convert_tokens_to_ids(None)` returned `None` silently. In 5.x it attempts to iterate over the argument and raises `TypeError: 'NoneType' object is not iterable`. This broke `build_model()` for backbones whose tokenizer has no `bos_token` (e.g. T5 sets `bos_token = None`).

**Fix:** Replaced `convert_tokens_to_ids(tokenizer.{pad,bos,eos}_token)` with the direct `tokenizer.{pad,bos,eos}_token_id` properties, which handle `None` correctly and are the idiomatic way to read these values.

---

### 18. `max_length` in model config rejected by `save_pretrained` in 5.x
**Files affected:** `tests/e2e_overfitting/config.yaml`

**Change:** In transformers 5.x, `PretrainedConfig.save_pretrained()` raises `ValueError` if any generation parameter (including `max_length`) is found in the model config: *"Some generation parameters are set in the model config. These should go into model.generation_config."* The e2e test config still carried `max_length: 10` under the `model:` section, which was a valid field in transformers 4.x.

**Fix:** Removed `max_length: 10` from the `model:` section of `tests/e2e_overfitting/config.yaml`. `max_length` was already removed from `MultiModalEmbedderConfig` in incompatibility #8; this removes the stale value from the test config that was silently stored as an extra kwarg until `save_pretrained` rejected it.

**How to set max_length going forward:**

Generation length is now controlled separately for each use case:

- **During training (eval/predict steps):** Set `generation_max_length` in the `training:` section of the YAML config. This maps to `Seq2SeqTrainingArguments.generation_max_length`, which the trainer stores in `_gen_kwargs` and passes directly to `model.generate()`.
  ```yaml
  training:
    generation_max_length: 128
  ```

- **During translation (`mmhugs-generate`):** Set `max_length` in the `generation:` section of the YAML config. This maps to `GenerateArgs.max_length` and is read in `translation_generate.py`.
  ```yaml
  generation:
    max_length: 128
  ```

- **Default behaviour when neither is set:** `model.generation_config.max_length` is `None` after setup (no value is baked in). `generate()` then applies a hardcoded fallback of **20 new tokens** (output length ≤ 21 including the decoder start token), with a UserWarning recommending `max_new_tokens` be set explicitly.

---

### 15. `_reorder_cache` removed from backbone models
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** `_reorder_cache` was removed from M2M100 and other backbone models in transformers 5.x. In the new cache system (`DynamicCache`), `GenerationMixin` calls `cache.reorder_cache(beam_idx)` directly on the cache object instead of delegating to the model.

**Fix:** Updated `MultiModalEmbedderModel._reorder_cache` to delegate to `past_key_values.reorder_cache(beam_idx)` when the cache object supports it (5.x path), and fall back to tuple-style reordering for legacy tuple-format caches (4.x compatibility).
