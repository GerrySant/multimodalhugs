# MultiLingualSeq2SeqTrainer — Design Notes

Tracks what `MultiLingualSeq2SeqTrainer` adds on top of the HuggingFace `Seq2SeqTrainer`, what
bugs exist, and what needs to be updated for `transformers==5.x` compatibility.

---

## Additions over HuggingFace Seq2SeqTrainer (baseline: transformers<=4.44.2)

### 1. Visualization helpers (custom feature — keep as-is)

Three extra `__init__` parameters:
- `visualize_prediction_prob: float = 0.05` — probability of logging a sample prediction each step
- `print_decoder_prompt_on_prediction: bool` — whether to print the decoder prompt portion
- `print_special_tokens_on_prediction: bool` — whether to include special tokens in printed output

And a `visualize_generation(preds, labels)` method that decodes and prints label/prediction pairs
during evaluation. This is a multimodalhugs-specific feature not present in the base class.

### 2. Variable-length decoder prompt handling (custom feature — keep as-is)

The 4.44.2 (and 5.x) base `Seq2SeqTrainer` assumes all samples in a batch can be passed together
to `model.generate()`. In multimodalhugs, different samples may have decoder prompts of different
lengths (e.g. different target language tokens). This is handled with a 3-way dispatch:

```python
if all_values_equal(generation_inputs['decoder_attention_mask']):
    # Uniform prompt lengths → batch generate (fast path)
    generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

elif generation_inputs['decoder_attention_mask'].numel() == 0:
    # No decoder prompt → strip prompt keys and batch generate
    generation_inputs.pop("decoder_input_ids", None)
    generation_inputs.pop("decoder_attention_mask", None)
    generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

else:
    # Variable-length prompts → generate sample by sample, then pad and concatenate
    ...
```

This logic is not in the HuggingFace base class and must be preserved.

---

## Known bugs

### Bug: `--generation_max_length` training arg is silently ignored

**Location:** `prediction_step`, line 130 (current):
```python
gen_kwargs = self.model.generation_config.to_dict()
```

**Problem:** This unconditionally populates `gen_kwargs` before the standard check:
```python
if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
    gen_kwargs = self._gen_kwargs.copy()
```
Since `gen_kwargs` is never empty after line 130, `_gen_kwargs` (which carries training args like
`--generation_max_length`) is never applied. The effective max_length is always whatever
`model.generation_config.max_length` holds.

**Fix:** Remove line 130 entirely. The 5.x base class handles generation config correctly through
the standard `_gen_kwargs` flow without this line.

---

## Updates required for transformers 5.x compatibility

### 1. `tokenizer` parameter renamed to `processing_class`

**Severity:** Breaking — will `TypeError` on construction.

In transformers 5.x, `Seq2SeqTrainer.__init__` (and `Trainer.__init__`) renamed the `tokenizer`
parameter to `processing_class`. `Trainer.__init__` sets `self.processing_class`, not
`self.tokenizer`.

**Changes required:**
- `__init__` signature: rename `tokenizer` parameter to `processing_class`
- `super().__init__()` call: `tokenizer=tokenizer` → `processing_class=processing_class`
- `visualize_generation()`: all `self.tokenizer` references → `self.processing_class`

### 2. Line 130 bug fix (also a 5.x update)

Remove `gen_kwargs = self.model.generation_config.to_dict()` (see bug section above).
In 5.x, the base class adds `gen_config._get_default_generation_params()` population logic, so
the generation config is handled correctly through the standard flow.

### 3. `synced_gpus` check missing FSDP

**Severity:** Low — only affects FSDP users.

Current code (line 146):
```python
default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
```

5.x base class:
```python
default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
```

**Fix:** Import `is_fsdp_managed_module` from `transformers.trainer_pt_utils` (or wherever 5.x
exposes it) and update the `synced_gpus` line to match the 5.x pattern.

---

## What is already in the 5.x base class (no longer needs to be in the override)

| Code | Status |
|---|---|
| `_from_model_config` hack (lines 196–197) | Now also in 5.x base — redundant but harmless |
| `gen_config._get_default_generation_params()` + `update(..., defaults_only=True)` | In 5.x base only — multimodalhugs should not need to replicate |
| `FullyShardedDataParallel.summon_full_params` context manager around `model.generate()` | In 5.x base only — multimodalhugs should not need to replicate |
