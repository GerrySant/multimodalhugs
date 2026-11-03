# Compatibility record — transformers 5

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

### 12. `CLIPModel._init_weights` crashes when text branch is nullified; replace with `CLIPVisionModelWithProjection`
**Files affected:** `multimodalhugs/modules/feature_extractor.py`

**Change:** In transformers 5.x, `from_pretrained` (and random-init via `post_init`) calls `initialize_weights() → smart_apply → _init_weights` on every submodule. `CLIPModel._init_weights` unconditionally accesses `module.text_projection.weight`, crashing with `AttributeError: 'NoneType' object has no attribute 'weight'` when the text branch has been nullified (`text_model = None`, `text_projection = None`).

**Root cause:** The old code loaded a full `CLIPModel` and discarded the text branch by setting `text_model = None` and `text_projection = None`. This worked in 4.x but is unsafe in 5.x because weight-init hooks run after submodule assignment and don't guard against `None` components.

**Fix:** Replaced `CLIPModel` + text nullification with `CLIPVisionModelWithProjection`:
- Vision-only model with no `text_model` or `text_projection` attributes whatsoever.
- Takes `CLIPVisionConfig`; extracted from the `CLIPConfig.vision_config` sub-config when a full `CLIPConfig` is passed.
- `forward(pixel_values=x).image_embeds` returns the projected vision features as a plain tensor — no `ModelOutput` unwrapping needed.
- `CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")` loads correctly from a full CLIP checkpoint (extra text-model keys are ignored with a warning).

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

### 15. `_reorder_cache` removed from backbone models
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** `_reorder_cache` was removed from M2M100 and other backbone models in transformers 5.x. In the new cache system (`DynamicCache`), `GenerationMixin` calls `cache.reorder_cache(beam_idx)` directly on the cache object instead of delegating to the model.

**Fix:** Updated `MultiModalEmbedderModel._reorder_cache` to delegate to `past_key_values.reorder_cache(beam_idx)` when the cache object supports it (5.x path), and fall back to tuple-style reordering for legacy tuple-format caches (4.x compatibility).

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

### 19. `Seq2SeqTrainer.__init__` no longer accepts `tokenizer` kwarg
**Files affected:** `multimodalhugs/multilingual_seq2seq_trainer.py`, `multimodalhugs/tasks/translation/translation_training.py`, `multimodalhugs/tasks/translation/translation_generate.py`

**Change:** In transformers 5.x, `Trainer.__init__` (and `Seq2SeqTrainer.__init__`) removed the `tokenizer` parameter. The replacement is `processing_class`, which accepts any processor/tokenizer instance.

**Fix:**
- Renamed `tokenizer` parameter to `processing_class` in `MultiLingualSeq2SeqTrainer.__init__`.
- Replaced `tokenizer=tokenizer` with `processing_class=tokenizer` at all three call sites.
- Replaced `self.tokenizer` with `self.processing_class` throughout the trainer body.

---

### 20. `generate()` requires `main_input_name` to identify the encoder input
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In transformers 5.x, `GenerationMixin.generate()` added a hard guard in `_prepare_model_inputs`: it looks up `self.main_input_name` in the provided kwargs to find the encoder input. If the named key is absent and `bos_token_id` is also None, it raises `ValueError: bos_token_id has to be defined when no input_ids are provided`. `PreTrainedModel.main_input_name` defaults to `"input_ids"`, but our model's primary encoder input is `"input_frames"`.

In 4.x the same check was absent, so `input_frames` stayed in `model_kwargs` and reached the encoder naturally through `_prepare_encoder_decoder_kwargs_for_generation`.

**Fix:** Added `main_input_name = "input_frames"` as a class attribute on `MultiModalEmbedderModel`. This is the standard HuggingFace pattern for models whose primary input is not token IDs (e.g. `CLIPModel.main_input_name = "pixel_values"`, `Wav2Vec2Model.main_input_name = "input_values"`). The `EncoderWrapper` already handles `input_frames` correctly; this was the missing declaration.

---

### 21. `GenerationConfig.max_length` defaults to `None` in 5.x; padding comparisons crash
**Files affected:** `multimodalhugs/multilingual_seq2seq_trainer.py`

**Change:** In transformers 4.x, `GenerationConfig.max_length` defaulted to 20. In 5.x it defaults to `None` (generation length is expected to be controlled via `max_new_tokens` or set explicitly). The custom `prediction_step` compared `generated_tokens.shape[-1] < self.generation_config.max_length` without a None guard, raising `TypeError: '<' not supported between instances of 'int' and 'NoneType'`.

**Fix:** Before the post-generation padding logic, apply `gen_config.update(**gen_config._get_default_generation_params(), defaults_only=True)`. This fills in `max_length=20` (and other defaults) for any field still set to `None`, matching what the 5.x `Seq2SeqTrainer.prediction_step` does internally. The `max_length` comparison then works correctly.

---

### 22. `synced_gpus` must include FSDP-managed models
**Files affected:** `multimodalhugs/multilingual_seq2seq_trainer.py`

**Change:** In transformers 5.x, `Seq2SeqTrainer.prediction_step` changed `default_synced_gpus` from `is_deepspeed_zero3_enabled()` to `is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)`. FSDP-sharded models require all ranks to stay in sync during autoregressive decoding, the same as DeepSpeed ZeRO-3.

**Fix:** Added `from transformers.integrations.fsdp import is_fsdp_managed_module` and updated the `default_synced_gpus` expression to match the 5.x parent.

---

### 23. Remaining `prediction_step` divergences from upstream 5.5 `Seq2SeqTrainer`
**Files affected:** `multimodalhugs/multilingual_seq2seq_trainer.py`

**Change:** After the fixes in #21 and #22, four additional divergences from the upstream
`Seq2SeqTrainer.prediction_step` in 5.5.0 remained.

**a) `synced_gpus` ternary → `.get()` idiom**

The assignment used a non-standard conditional expression:
```python
gen_kwargs["synced_gpus"] = (
    gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
)
```
Upstream uses the standard dict `.get()` with a default:
```python
gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)
```

**b) Missing `_from_model_config` reset**

Upstream 5.5 resets `generation_config._from_model_config = False` immediately after the
generation block to prevent the config from being re-initialized on every eval iteration:
```python
if self.model.generation_config._from_model_config:
    self.model.generation_config._from_model_config = False
```
Our override was missing this block entirely.

**c) Spurious `self.generation_config = gen_config` assignment**

After `gen_config.update(...)`, the old code wrote:
```python
self.generation_config = gen_config
```
Upstream does not do this. Storing a mutable reference to the generation config on the trainer
instance is a side effect that can cause config state to bleed between eval iterations. Removed.

**d) Stale `generation_max_length` padding checks + wrong check order**

`Seq2SeqTrainer.generation_max_length` no longer exists in 5.5. The old code had:
```python
if hasattr(self, "generation_max_length") and ...:
    generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.generation_max_length + 1)
elif gen_config.max_new_tokens is not None and ...:
    ...
elif generated_tokens.shape[-1] < gen_config.max_length:
    ...
```
When `generation_max_length` was absent (always in 5.5), the `max_new_tokens` branch ran before
the `max_length` branch — the opposite order from upstream. Both the stale check and the wrong
order were corrected to match 5.5:
```python
if generated_tokens.shape[-1] < gen_config.max_length:
    generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
elif gen_config.max_new_tokens is not None and ...:
    generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
```
The same correction was applied to the `labels` padding block.

**e) `.mean().detach()` → `.detach().mean()`**

Upstream 5.5 detaches the loss tensor before reducing to avoid retaining unnecessary gradient
graphs in memory during evaluation:
```python
# Old:
loss = (...).mean().detach()
# New (upstream 5.5):
loss = (...).detach().mean()
```
Applied to both the `label_smoother` path and the default `outputs["loss"]` path.

---

### 24. Manual `_no_split_modules`/`_keep_in_fp32_modules` propagation is redundant in 5.5
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In transformers 5.5, `PreTrainedModel.post_init()` was significantly enhanced to
automatically propagate `_no_split_modules`, `_keep_in_fp32_modules`, and `all_tied_weights_keys`
from all `named_children()` that are themselves `PreTrainedModel` instances. Since `backbone` and
`feature_extractor` are both `PreTrainedModel` subclasses and direct children of
`MultiModalEmbedderModel`, `post_init` already lifts their properties to the outer model.

The manual propagation in `_init_backbone` and `_init_feature_extractor` (list concatenation with
`list(getattr(...))`) was therefore doing the same work twice. Worse: the manual code produced
lists while `post_init` expects sets (it calls `.update()` on them), creating a latent
type-inconsistency between the two passes.

**Fix:** Removed the manual `_no_split_modules` and `_keep_in_fp32_modules` propagation lines from
both `_init_backbone` and `_init_feature_extractor`. The `is_parallelizable` propagation was kept
because `post_init` in 5.5 does not handle that attribute automatically.

---

### 25. `model.backbone.tie_weights()` should be `model.tie_weights()` after `load_state_dict`
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Change:** In `build_model()`, after copying weights with `load_state_dict`, the code called
`model.backbone.tie_weights()` to reconnect tied parameters. In transformers 5.5, `post_init`
computes `all_tied_weights_keys` at the *composite* model scope — the keys are stored as
`backbone.lm_head.weight → backbone.model.shared.weight` (prefixed with `backbone.`). Calling
`model.backbone.tie_weights()` re-ties within the backbone's own scope using backbone-local paths,
which also works, but it bypasses the composite-level bookkeeping that 5.5 established.

**Fix:** Changed `model.backbone.tie_weights()` to `model.tie_weights()`. This is the semantically
correct call that aligns with how `from_pretrained` re-ties weights in 5.5 and operates at the
right scope for a composite model.

---

### 26. `AutoModelForVision2Seq` removed — register with `AutoModelForImageTextToText`
**Files affected:** `multimodalhugs/models/__init__.py`

**Change:** `AutoModelForVision2Seq` was removed in transformers 5.5 and replaced by
`AutoModelForImageTextToText`. For vision-to-text models (image, video, pose, SignWriting → text),
`AutoModelForImageTextToText` is the semantically correct auto class in 5.5.

**Fix:** Added `AutoModelForImageTextToText.register(config_class, model_class)` alongside the
existing `AutoModelForSeq2SeqLM.register()` in the model registration loop. Both registrations
are kept — `AutoModelForSeq2SeqLM` for backward compatibility with existing code that loads via
that class, and `AutoModelForImageTextToText` for correct HF ecosystem integration in 5.5.

---

### 27. `VideoModalityProcessor` migrated to `transformers.video_utils.load_video()`
**Files affected:** `multimodalhugs/processors/video_modality_processor.py`

**Change:** The internal `_load_video` implementation previously used two separate code paths:
cv2 (when `custom_preprocessor_path` was set) and `torchvision.read_video` (otherwise). Both
were hand-rolled (~120 lines) and tightly coupled — cv2 was required whenever a custom
preprocessor was used, regardless of whether cv2 was actually needed.

Replaced with `transformers.video_utils.load_video()` (new in transformers 5.5), which provides
a unified interface over five backends: `"pyav"`, `"torchvision"`, `"decord"`, `"opencv"`, and
`"torchcodec"`. Clip extraction (signal_start/signal_end in ms or frame units) is expressed as a
`sample_indices_fn` closure passed to `load_video()`.

**New parameters:**
- `backend: str = "pyav"` — explicit backend selection, independent of `custom_preprocessor_path`.
- `num_frames: Optional[int] = None` — uniform frame subsampling via `np.linspace`. See below.

**Frame sampling: `num_frames` vs `skip_frames_stride`**

Two parameters control how many frames are extracted from a clip window:

- `skip_frames_stride=N` (pre-existing): keeps every N-th frame via
  `np.arange(start_frame, end_frame, N)`. Output length scales with clip length — a 50-frame clip
  with stride 2 gives 25 frames; a 100-frame clip gives 50. Use when variable-length sequences
  are acceptable and you want proportional frame-rate reduction.

- `num_frames=N` (new): uniformly samples exactly N frames via
  `np.linspace(start_frame, end_frame - 1, N, dtype=int)`. Output is always exactly N frames
  regardless of clip length. Use when the model expects a fixed temporal dimension.

**Precedence when both are set:**

`num_frames` wins **only when `num_frames < clip_length`** (the `linspace` branch runs). If the
clip is shorter than `num_frames` (e.g. a 10-frame clip with `num_frames=16`), the condition is
false and execution falls through to `skip_frames_stride` if set, or all frames if neither
condition is met. Setting both is therefore redundant in the typical case and potentially
surprising for unusually short clips. Recommendation: pick one.

**Removed:** Module-level `_CV2_AVAILABLE` and `_TORCHVISION_AVAILABLE` import guards. Backend
availability is now deferred to call time; `load_video()` raises a clear `ImportError` if the
chosen backend is not installed.

**torchcodec and the fork/spawn constraint:**

`backend="torchcodec"` decodes video directly to CUDA tensors, eliminating the CPU→GPU copy
bottleneck. However, on Linux, PyTorch DataLoader defaults to the `"fork"` multiprocessing start
method. A process that has an active CUDA context cannot safely be forked — worker processes
inherit a broken CUDA state, causing errors or deadlocks when `num_workers > 0`.

`VideoModalityProcessor.__init__` detects this at construction time: if `backend="torchcodec"` is
used and the effective start method is `"fork"` (either explicitly set or implied by the Linux
default), a `logger.warning` is emitted telling the user to call
`torch.multiprocessing.set_start_method("spawn")` or set `dataloader_num_workers=0`.

HuggingFace transformers 5.5 does not add this guard anywhere in its own video processing code
(`video_utils.py`, `trainer.py`) — users are expected to know the general CUDA+fork rule.
multimodalhugs adds the warning explicitly to catch misconfiguration before training starts.

**torchcodec + custom_preprocessor_path:**

With transformers 5.x, the default `TorchvisionBackend` image processor accepts tensors directly
and runs on the same device as the input. When `backend="torchcodec"`, `device="cuda"`, and
`custom_preprocessor_path` is set, the decoded CUDA tensor is passed straight to the preprocessor
without any CPU transfer — the full zero-copy GPU decode + preprocess pipeline is available.
See entry 28 for details on the implementation change that enabled this.

**Configuring spawn from YAML:**

The `worker_start_method` field in `ExtendedSeq2SeqTrainingArguments` (a multimodalhugs extension
to `Seq2SeqTrainingArguments`) exposes this as a YAML config option. It is applied early in
`translation_training.py::main()`, before any CUDA or multiprocessing initialization:

```yaml
training:
  dataloader_num_workers: 4
  worker_start_method: spawn
```

This is a global setting — it affects DataLoader workers, `Dataset.map()` workers, and all other
child processes spawned via `torch.multiprocessing`.

**Correction — torchcodec device defaults to CPU:**

`read_video_torchcodec` in transformers 5.5 uses `device=kwargs.get("device", "cpu")`, so
torchcodec decodes to CPU by default. GPU decode (NVDEC) requires explicitly passing
`device="cuda"`. `VideoModalityProcessor` exposes this via a `device` parameter (default `None`,
which leaves the backend's default intact). Setting `device="cuda"` in the processor config
activates true GPU decode. The fork/spawn warning only fires when `device` is a CUDA device.

**Correction 2 — auto-detect removed:**

An earlier version of `VideoModalityProcessor.__init__` auto-detected the current CUDA device
when `backend="torchcodec"` and `device=None`: it called `torch.cuda.current_device()` in the
main process and silently upgraded `device` to `"cuda:N"`. This had two problems: (1) the
detected device is the main process's device at construction time — DataLoader workers may run
on a different device in multi-GPU setups; (2) there was no way to use torchcodec on CPU when
CUDA is available without explicitly passing `device="cpu"`. The auto-detect block was removed.
`device=None` now means exactly what the docstring says: leave torchcodec's own default intact
(CPU). Users who want GPU decode must set `device="cuda"` explicitly.

---

### 28. `load_video()` return shapes: torchvision also returns a tensor (not numpy)
**Files affected:** `multimodalhugs/processors/video_modality_processor.py` (comment-only)

**Finding:** The inline comments in `_load_video_impl` described the return shapes of
`load_video()` as "numpy for most backends; tensor for torchcodec". This was incomplete.
Verified from `transformers/video_utils.py` source:

| Backend    | Return type      | Shape         |
|------------|-----------------|---------------|
| pyav       | `np.ndarray`    | `[T, H, W, C]` uint8 |
| opencv     | `np.ndarray`    | `[T, H, W, C]` uint8 |
| decord     | `np.ndarray`    | `[T, H, W, C]` uint8 |
| torchvision | `torch.Tensor` | `[T, C, H, W]` uint8 (deprecated in 5.5) |
| torchcodec | `torch.Tensor`  | `[T, C, H, W]` uint8 |

torchvision uses `output_format="TCHW"` in `torchvision.io.read_video`, so it also returns a
channels-first tensor. The `isinstance(frames, torch.Tensor)` branch in `_load_video_impl`
correctly handles both torchvision and torchcodec. No functional change was needed; comments
were updated to name both tensor-returning backends explicitly.

---

### 29. `TorchvisionBackend` (transformers 5.x) enables full GPU preprocessing pipeline
**Files affected:** `multimodalhugs/processors/video_modality_processor.py`

**Change:** In transformers 5.x, the default image processor backend is `TorchvisionBackend`
(torchvision must be installed; falls back to `PilBackend` otherwise). Unlike the 4.x
`CLIPImageProcessor` (PIL-only), `TorchvisionBackend`:
- Accepts `torch.Tensor`, `np.ndarray`, or `PIL.Image` inputs interchangeably.
- "By default, the processing is done on the same device as the inputs if the inputs are
  tensors." — so passing a CUDA tensor runs resize/normalize on GPU with no CPU transfer.
- Exposes a `device` argument on the `__call__` to override the auto-detected device.

**Old code** (pre-28) always converted tensor frames to CPU PIL images before calling the
preprocessor:
```python
pil_frames = [Image.fromarray(frames[i].cpu().permute(1, 2, 0).numpy()) ...]
self.custom_preprocessor(images=pil_frames, return_tensors="pt")
```

**New code** passes tensor frames directly and forwards `self.device` to the preprocessor:
```python
input_frames = list(frames)  # list of [C, H, W] tensors, stay on original device
call_kwargs = {"images": input_frames, "return_tensors": "pt"}
if self.device is not None:
    call_kwargs["device"] = self.device
result = self.custom_preprocessor(**call_kwargs)["pixel_values"]
```

For numpy-returning backends (pyav, opencv, decord), PIL conversion is kept as before.

**Result:** With `backend="torchcodec"`, `device="cuda"`, and `custom_preprocessor_path` set,
the complete pipeline — decode, resize, normalize — runs on GPU with zero CPU involvement.
For CPU-decoded backends with `device="cuda"`, the preprocessor moves frames to GPU for the
processing step (one CPU→GPU transfer, then GPU resize/normalize).

---

### 30. `FeatureExtractor.__init__` called `from_pretrained` inside meta-device context
**Files affected:** `multimodalhugs/modules/feature_extractor.py`, `multimodalhugs/modules/__init__.py`, `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Error:** `RuntimeError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.Tensor.to(device) when moving module from meta to a different device.` (surface message varies by torch version; underlying cause is calling `from_pretrained` while inside `init_empty_weights()`).

**Root cause:** In transformers 5.x, `from_pretrained` uses `init_empty_weights()` (meta-device context) to allocate the model structure without real memory before loading the checkpoint. Any call to another `from_pretrained` inside `__init__` while in this context is forbidden — meta tensors cannot be copied or moved.

The old `FeatureExtractor.__init__` called `from_pretrained` directly when `pretrained_module` was set:
```python
if pretrained_module is not None:
    self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained(pretrained_module)
```
This ran fine at build time (no meta-device context) but crashed when `MultiModalEmbedderModel.from_pretrained(checkpoint)` later tried to reconstruct the model structure.

**Fix:**
- `FeatureExtractor.__init__` is now **structure-only**: it allocates the inner model from a config but never calls `from_pretrained`.
- Added `load_pretrained_feature_extractor(feature_extractor_type, pretrained_module)` as a module-level function in `feature_extractor.py`. It contains all FE-type-specific loading logic (e.g. `CLIPVisionModelWithProjection` for `"clip"`), keeping it out of the model script and making it easy to extend for new FE types.
- `build_model` now: (1) calls `load_pretrained_feature_extractor` **before** `cls(config=cfg)` to load the FE outside any meta-device context; (2) stores the FE's config in `cfg.feature_extractor_config` so `FeatureExtractor.__init__` can reconstruct the same structure; (3) copies weights with `model.feature_extractor.feature_extractor.load_state_dict(fe_model.state_dict())` after the model is built.
- When loading from a saved checkpoint via `from_pretrained`, the FE weights are loaded by the normal HF mechanism from the combined checkpoint — no special handling is needed.

**Why it worked in 4.x:** transformers 4.x `from_pretrained` allocated random weights and overwrote them with the checkpoint. `__init__` ran normally (no meta-device context), so nested `from_pretrained` calls inside `__init__` worked. In 5.x the workflow is inverted: structure first (on meta device), then weights — which is why `__init__` must be weight-free.

---

### 31. `_init_backbone` received `backbone_config` as dict instead of `PretrainedConfig`
**Files affected:** `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`

**Root cause:** `PretrainedConfig` subclasses are serialised as plain Python dicts in `config.json`. When `MultiModalEmbedderModel.from_pretrained(checkpoint)` loads the config, `config.backbone_config` can arrive as a raw dict (e.g. if the `MultiModalEmbedderConfig.__init__` conversion fails or is bypassed). Passing a dict directly to `BackboneModelClass(backbone_cfg)` raises `TypeError` because model constructors expect a typed `PretrainedConfig` instance.

This was a latent bug introduced during the 5.x migration: the old code used `from_pretrained` unconditionally for the backbone, so `_init_backbone` never passed a config object directly. The new config-based branch was added without a dict guard.

**Fix:** Added an `isinstance(backbone_cfg, dict)` check in `_init_backbone` before calling the backbone constructor:
```python
if isinstance(backbone_cfg, dict):
    BackboneConfigClass = get_backbone_config_class(config.backbone_type)
    backbone_cfg = BackboneConfigClass.from_dict(backbone_cfg)
self.backbone = BackboneModelClass(backbone_cfg)
```
`from_dict` correctly reconstructs the typed config from the serialised dict, including handling extra or version-specific fields gracefully.

---

### 32. `merge_config_and_command_args` returned stale `_args`, silently disabling FP16/AMP
**Files affected:** `multimodalhugs/tasks/translation/utils.py`

**Symptom:** Training speed ~60% slower than transformers 4.x on identical hardware and config (~17–18 s/it vs ~9–12 s/it on A100) despite `fp16: true` in the YAML. All matrix multiplications run in FP32 (`ampere_sgemm_*` kernels instead of `ampere_fp16_s16816gemm_*`). Diagnostic: `trainer.accelerator.mixed_precision` is `"no"` and `trainer.accelerator.scaler` is `None`.

**Root cause:** In transformers 5.x, `Trainer` delegates all mixed-precision handling to Accelerate. It reads `training_args.mixed_precision` (a derived attribute computed in `TrainingArguments.__post_init__` from `fp16`/`bf16`) and passes it to `Accelerator(mixed_precision=...)`. The attribute is computed once at instance creation time and is never recomputed.

`merge_config_and_command_args` correctly created a fresh `extra_args` instance via `HfArgumentParser.parse_dict()` — this triggered `__post_init__`, which set `extra_args.mixed_precision = "fp16"`. However, the function then called `merge_arguments(cmd_args=_args, ...)`, which patched the original CLI-parsed `_args` via `setattr` and returned that stale instance. `_args.mixed_precision` was never updated: Accelerate received `mixed_precision="no"` and disabled AMP entirely.

This was invisible in transformers 4.x because that version read `training_args.fp16` directly (a primary field set correctly by `setattr`) and called `torch.cuda.amp.autocast()` itself — `mixed_precision` was never consulted.

**Fix:** Removed the `merge_arguments` call. Instead, CLI-explicit fields are re-applied directly onto `extra_args` (to preserve CLI precedence over YAML), and `extra_args` is returned:
```python
for f in fields(extra_args):
    if f.name in command_arg_names:
        setattr(extra_args, f.name, getattr(_args, f.name))
return extra_args
```
`extra_args` was created by `parse_dict`, so `__post_init__` has already run and all derived attributes (`mixed_precision`, `half_precision_backend`, `parallel_mode`, etc.) are correctly computed.

---

### 33. `add_special_tokens({'extra_special_tokens': [...]})` wipes M2M100 language codes
**Files affected:** `multimodalhugs/utils/tokenizer_utils.py`

**Symptom:** After `multimodalhugs-setup` with `new_vocabulary` configured (e.g. `new_vocabulary: "__asl__"`), `batch_decode(..., skip_special_tokens=True)` no longer strips language tokens like `__en__` during decoding. Both predictions and labels are returned with `__en__` prepended, inflating BLEU and chrF metrics. Example observable in training logs: `Prediction - __en__ So the juice is made out of oil...`. The inflation can be dramatic — a real run showed sacrebleu rise from ~9.56 (correct) to ~23.81 (inflated).

**Root cause:** Introduced by commit `b59666b` ("[fix] Update tokenizer special-tokens API for transformers 5.x") which correctly removed the deprecated `replace_additional_special_tokens` kwarg but incorrectly changed the key from `additional_special_tokens` to `extra_special_tokens` for all tokenizer types.

In transformers 5.x, `M2M100Tokenizer` (a slow tokenizer) stores its 100 language codes in an internal attribute `_extra_special_tokens`. Calling `add_special_tokens({'extra_special_tokens': [new_token]})` **replaces** `_extra_special_tokens` entirely — the old language codes are gone. The same key that `PreTrainedTokenizerFast` uses for extending its special tokens is a destructive replacement on `M2M100Tokenizer`. Neither key is safe to use on M2M100 without first preserving the existing list.

After the call, the serialized `tokenizer_config.json` has `"additional_special_tokens": []` and `"extra_special_tokens": ["__asl__"]`. On reload, `all_special_tokens` contains only 5 entries (4 standard + `__asl__`) instead of 105, so `__en__` is no longer stripped during decoding.

In transformers 4.x this code path was never reached: `add_special_tokens({'extra_special_tokens': [...]})` raised `AssertionError: Key extra_special_tokens is not a special token` — the key did not exist. The migration commit introduced the key and the replacement behavior simultaneously.

**Fix:** Read `_extra_special_tokens` before the call and pass the full merged list:
```python
existing_extra = [str(t) for t in getattr(tokenizer, '_extra_special_tokens', [])]
all_extra = existing_extra + [t for t in added_tokens if t not in existing_extra]
tokenizer.add_special_tokens({'extra_special_tokens': all_extra})
```
`getattr(..., [])` is a no-op for tokenizers that do not have `_extra_special_tokens` (e.g. `PreTrainedTokenizerFast`). For `M2M100Tokenizer`, it preserves all language codes and appends only the new tokens.

**Affected experiments:** Any experiment satisfying all of: M2M100-based backbone, `new_vocabulary` configured, language token used as `decoder_prompt`/`target_prefix`, run with transformers 5.x after commit `b59666b`. Already-saved processor directories must be regenerated with `multimodalhugs-setup` after applying this fix.

---

### 34. `overwrite_output_dir` removed from `TrainingArguments` in transformers 5.x
**Files affected:** `multimodalhugs/tasks/translation/translation_training.py`

**Change:** `TrainingArguments.overwrite_output_dir` was removed in transformers 5.x. The 4.x field controlled whether the Trainer would raise `ValueError` when writing into a non-empty output directory. In 5.x, output-dir conflict resolution is handled internally by the Trainer.

**Root cause:** `translation_training.py` copied the standard checkpoint-detection block from HF's `run_translation.py` template, which included a `not training_args.overwrite_output_dir` guard. In 5.x `training_args.overwrite_output_dir` raises `AttributeError`.

**Fix:** Removed the `not training_args.overwrite_output_dir` guard and the `ValueError` that fired on non-empty output directories. Checkpoint detection (`get_last_checkpoint`) still runs unconditionally when `do_train=True` and the output dir exists — only the protection against overwriting was removed.

---

### 35. `data_args.max_source_length` activated by our own `__getattr__` fix (entry 14)
**Files affected:** `multimodalhugs/tasks/translation/translation_training.py`

**Change:** `DataTrainingArguments` in multimodalhugs never had a `max_source_length` field (HF's upstream template had it but we didn't add it). The check:
```python
if model.config.max_position_embeddings < data_args.max_source_length:
```
was always dead code — `hasattr(model.config, "max_position_embeddings")` returned `False` in 4.x (before `__getattr__` was added to `MultiModalEmbedderConfig`), so the short-circuit prevented `data_args.max_source_length` from ever being accessed.

Entry 14 added `__getattr__` delegation to `MultiModalEmbedderConfig` so that `GenerationMixin` can find `vocab_size`, `hidden_size`, etc. via `backbone_config`. This same delegation makes `hasattr(model.config, "max_position_embeddings")` return `True` for M2M100-based models (M2M100Config has the attribute). The short-circuit no longer fires and `data_args.max_source_length` is accessed — raising `AttributeError` because the field was never in our dataclass.

**Fix:** Removed the entire positional-embedding length check. It was never valid for multimodalhugs (wrong dataclass field) and the underlying concern (source length exceeding model capacity) does not apply to multimodalhugs use cases, where encoder inputs are variable-length frame sequences rather than token sequences bounded by `max_position_embeddings`.

---

### 36. `decoder_input_ids` slot renamed to `decoder_prompt_ids`; train/inference prefix mismatch fixed
**Files affected:** `multimodalhugs/training_setup/setup_utils.py`, `multimodalhugs/data/datacollators/multimodal_datacollator.py`, `multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py`, `multimodalhugs/multilingual_seq2seq_trainer.py`

**Bug 1 — Silent train/inference prefix mismatch (always present):**

During training, `prepare_decoder_input_ids_from_labels` (shift-right) prepends `decoder_start_token_id` (e.g. `</s>`=2 for M2M100) so the model always sees `[</s>, __en__, tok1, …]` at decoder position 0. During inference, the `decoder_input_ids` processor slot tokenised the raw `decoder_prompt` column and passed `[__en__]` directly to `model.generate()` — a prefix the model was never trained on. Verified: training position 0 = `</s>` (id 2), inference position 0 = `__en__` (id 128022).

**Bug 2 — `label_smoothing_factor` race condition (crash when `eval_on_start=True`):**

`DataCollatorMultimodalSeq2Seq.__call__` only called `prepare_decoder_input_ids_from_labels` when `self.model.training is True`. After `eval_on_start` evaluation, DataLoader workers inherited `model.training=False`. The first training batch received the raw 1-token prompt as `decoder_input_ids` instead of the full shifted-labels sequence. With `label_smoothing_factor > 0`, `LabelSmoother` compared logits `[B,1,vocab]` against labels `[B,T]` and crashed: `RuntimeError: Size does not match at dimension 1 expected index [B, T, 1] to be no larger than self [B, 1, vocab_size]`.

**Fix — four coordinated changes:**

`setup_utils.py` — `_TEXT_SLOTS_TEMPLATE` slot 3: renamed `output_data_key` `decoder_input_ids` → `decoder_prompt_ids` and `output_mask_key` `decoder_attention_mask` → `decoder_prompt_mask`. The slot now carries only the raw per-sample generation prefix. The collator always builds teacher-forcing `decoder_input_ids` from labels independently.

`multimodal_datacollator.py`: removed `and self.model.training` from both the `MetaProcessor` and legacy paths in `DataCollatorMultimodalSeq2Seq` so `prepare_decoder_input_ids_from_labels` runs unconditionally.

`modeling_multimodal_embedder.py`: added `decoder_prompt_ids` and `decoder_prompt_mask` as `None`-default parameters to `MultiModalEmbedderModel.forward()`. `remove_unused_columns=False` means every batch key reaches `model(**inputs)` unchanged; without this the renamed keys would raise `TypeError`. The parameters are accepted and immediately ignored — no logic change.

`multilingual_seq2seq_trainer.py` — `prediction_step`: replaced the old dispatch block (which unconditionally accessed `generation_inputs['decoder_attention_mask']` after the eviction step, raising `KeyError`). New logic pops `decoder_prompt_ids`/`decoder_prompt_mask`, reads `bos = model.config.decoder_start_token_id` (backbone-specific; `</s>`=2 for M2M100/BART, `<pad>`=0 for T5, lang_id for mBART-cc25), and dispatches to: (a) empty-prompt path — `generate()` without `decoder_input_ids`; (b) uniform-prompt path — batched generation with prefix `[bos, prompt_tokens]`; (c) mixed-prompt path — sample-by-sample generation for variable-length prompts.

**`bos` computation — `or` replaced with explicit `None` check:**
The original design used Python's `or` to fall back from config to generation_config:
```python
bos = self.model.config.decoder_start_token_id or self.model.generation_config.decoder_start_token_id
```
For T5/ByT5 `decoder_start_token_id = 0` is falsy, so `0 or x` evaluates to `x`. If
`generation_config.decoder_start_token_id` were also `None`, `bos` would become `None` and the
batched prefix path would crash. Fixed with an explicit `None` check:
```python
_bos = self.model.config.decoder_start_token_id
if _bos is None:
    _bos = self.model.generation_config.decoder_start_token_id
bos = _bos
```
This is safe for `decoder_start_token_id = 0` (T5/ByT5) and handles the `None` fallback correctly
for all backbone families. Note: for T5/ByT5 with an empty `decoder_prompt` (the correct setup
per the backbone guide), `bos` is computed but never used — the empty-prompt dispatch branch fires
before any prefix construction.

**Note:** configs using the `slots:` format (instead of the `pipeline:` shorthand) must also rename `output_data_key: decoder_input_ids` → `decoder_prompt_ids` and `output_mask_key: decoder_attention_mask` → `decoder_prompt_mask` in their YAML, and rebuild the processor with `multimodalhugs-setup --do_processor`.

---

### 37. `ImageModalityProcessor` migrated to HF ecosystem and `TorchvisionBackend` (parity with `VideoModalityProcessor`)
**Files affected:** `multimodalhugs/processors/image_modality_processor.py`, `tests/test_data/test_optional_import_guards.py`, `tests/test_data/test_processor_image2text.py`, `docs/processors/image_modality_processor.md`

This entry covers four coordinated commits that brought `ImageModalityProcessor` to the same level as `VideoModalityProcessor` with respect to the transformers 5.x `TorchvisionBackend`. The changes mirror entry 27 (video `load_video` migration) and entry 29 (video GPU pipeline).

**a) Replace `cv2` with `transformers.image_utils.load_image()`**

`_load_from_path` previously used `cv2.imread`, which reads images in BGR and requires `opencv-python` as a hard dependency for basic image loading. Replaced with `transformers.image_utils.load_image()`, which:
- Returns a PIL RGB image with EXIF rotation already applied (no manual BGR→RGB swap).
- Accepts local file paths, HTTP/HTTPS URLs, and base64-encoded strings without extra configuration.
- Is always available as a transitive dependency of transformers; `opencv-python` is no longer required for the image-loading path.

Removed: the module-level `_CV2_AVAILABLE` import guard and the `__init__`-time `ImportError` raised when cv2 was absent.

**b) Add `custom_preprocessor_path` parameter**

Added `custom_preprocessor_path` (HuggingFace model ID or local path) to `__init__`. When set, loaded images and rendered text frames are passed through the specified HuggingFace image processor (e.g. `CLIPImageProcessor`), which handles resize, normalisation, and channel reordering. This makes `ImageModalityProcessor` consistent with `VideoModalityProcessor`, which already supported `custom_preprocessor_path` via `AutoProcessor`.

Setting `custom_preprocessor_path` takes precedence over `normalize_image`/`mean`/`std`; a `logger.warning` is emitted if both are set.

**c) Fix output shape to `[1, C, H, W]`**

Image files loaded via `_load_from_path` previously returned `[H, W, C]` (channels last, no temporal dimension). This was incompatible with the CLIP `FeatureExtractor` in the model (which expects `[B, T, C, H, W]`) and inconsistent with text-rendered images (`[N_words, C, H, W]`) and video frames (`[T, C, H, W]`).

Fix:
- Without `custom_preprocessor_path`: `[H, W, C]` numpy → `permute(2,0,1)` → `[C, H, W]` → `unsqueeze(0)` → `[1, C, H, W]`.
- With `custom_preprocessor_path`: `CLIPImageProcessor` already returns `[1, C, H, W]`; the `squeeze(0)` that was producing `[C, H, W]` is removed.

`process_batch` no longer needs a special branch for `custom_preprocessor`: `pad_and_create_mask` on `[1, C, H, W]` samples gives `[B, 1, C, H, W]` with a `[B, 1]` mask, consistent with video single-frame batches.

**d) Apply `custom_preprocessor` to text-rendered images**

`custom_preprocessor_path` was only applied in `_load_from_path` (image files/URLs); text-rendered images returned by `_render_text` bypassed it entirely. After rendering, `process_sample` now converts the `[N_words, C, H, W]` float32 frames to a list of uint8 PIL images and passes them through the preprocessor, producing `[N_words, C', H', W']` in the preprocessor's normalised range. `_render_text` also skips `normalize_image` when `custom_preprocessor` is set to prevent double-normalisation.

All four combinations now produce consistent `[T, C, H, W]` output:
- text, no preprocessor → `[N_words, 3, H, W]`
- text, with preprocessor → `[N_words, 3, 224, 224]`
- file, no preprocessor → `[1, 3, H, W]`
- file, with preprocessor → `[1, 3, 224, 224]`

**e) Add `device` parameter for GPU-accelerated preprocessing**

Added `device: Optional[str] = None` to `__init__`. When set, it is forwarded to the `custom_preprocessor.__call__()` as the `device` keyword argument in both the image-file path (`_load_from_path`) and the text-rendered path (`process_sample`):
```python
call_kwargs = {"images": ..., "return_tensors": "pt"}
if self.device is not None:
    call_kwargs["device"] = self.device
result = self.custom_preprocessor(**call_kwargs)["pixel_values"]
```
In transformers 5.x, the default `TorchvisionBackend` honours this argument and runs resize/normalise on the specified device, enabling GPU-accelerated image preprocessing without any explicit `.to(device)` call in user code. When `device=None` (default) the preprocessor runs on CPU. Ignored when `custom_preprocessor_path` is not set.

This is the exact same pattern as entry 29's `VideoModalityProcessor` implementation.

---

### 38. `TrainingArguments._validate_args()` strict hardware check in transformers 5.x
**Files affected:** `tests/test_data/test_translation_utils.py`, `tests/TESTS.md`

**Change:** In transformers 5.x, `TrainingArguments._validate_args()` raises `ValueError: Your setup doesn't support bf16/gpu` when `bf16=True` (or `fp16=True`) is set on a CPU-only machine or a machine without BF16-capable hardware. In transformers 4.x this check either did not exist or was not enforced at instantiation time.

`test_bf16_from_yaml_sets_mixed_precision` verifies that `merge_config_and_command_args` returns a freshly constructed `ExtendedSeq2SeqTrainingArguments` instance (so `__post_init__` has run and `mixed_precision="bf16"` is set correctly). The test does not test hardware BF16 support, but on CPU-only CI runners the strict 5.x check fires during `__post_init__` and raises before the assertion is reached.

**Fix:** Wrap the `merge_config_and_command_args` call in:
```python
with patch.object(ExtendedSeq2SeqTrainingArguments, "_validate_args"):
    result = merge_config_and_command_args(...)
```
Patching directly on the concrete class ensures the mock is found first in the MRO regardless of any `_validate_args` overrides in intermediate classes across transformers versions. `mixed_precision` is computed by `__post_init__` before `_validate_args` is called, so the patch suppresses only the hardware-check exception without affecting the attribute being asserted.
