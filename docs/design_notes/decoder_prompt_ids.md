# Decoder prompt carried as `decoder_prompt_ids`, not `decoder_input_ids`

Status: accepted
Date: 2026-06-04 (commit aa9aca7)

## Context

The processor slot that tokenises the `decoder_prompt` column (for example `__en__`, the
M2M100 target-language code) wrote its output to the batch key `decoder_input_ids`, the key
the model also uses for teacher forcing. One key carried two meanings, which caused two bugs.

Train/inference prefix mismatch (silent, present since the slot existed). In training the
collator builds `decoder_input_ids` by shifting the labels right, so the decoder sees
`[</s>, __en__, tok1, …]`: position 0 is `decoder_start_token_id` (`</s>` = 2 for M2M100).
At inference the slot output `[__en__]` was passed straight to `generate()`, so position 0
was the language token, a prefix the model was never trained on.

Label-smoothing race on `model.training` (crash). To keep the one-token prompt intact for
generation, the collator replaced it with the shifted labels only while `model.training` was
true. After `eval_on_start` (or any evaluation) DataLoader workers hold a copy of the model
with `training=False`, so the first training batch (every batch of the epoch when
`dataloader_num_workers > 0`) reached the model with the one-token prompt. With
`label_smoothing_factor > 0` the Trainer pops `labels` before `forward`, the model returns
logits `[B, 1, V]`, and `LabelSmoother` crashes against labels `[B, T]`.

## Options considered

HF-style conditioning: drop the decoder-prompt slot, always collate shifted labels, and
select the target language with `GenerationConfig.forced_bos_token_id`, as HF seq2seq models
do. Cleanest alignment, but `forced_bos_token_id` is one value per `generate()` call, so
mixed-language batches and multi-token prompts, both supported today, would be lost, and
every config and saved processor would need a new field.

Keep one key, fix the race: keep the slot writing `decoder_input_ids` and replace the
`model.training` check with an explicit per-sample train/eval flag set by the training
script. Smallest diff, but the two meanings stay conflated and the prefix mismatch remains.

Two keys (chosen): the slot writes the raw prompt to `decoder_prompt_ids` /
`decoder_prompt_mask`; the collator always builds `decoder_input_ids` from the labels, as
HF's `DataCollatorForSeq2Seq` does; `prediction_step` builds the generation prefix as
`[decoder_start_token_id] + prompt`. Costs a slot rename and two accepted-and-ignored
parameters on `forward()`.

## Decision

Two keys, implemented in aa9aca7 (`compat_transformers_5.md`, entry 36). The pipeline
shorthand's third text slot now emits `decoder_prompt_ids` / `decoder_prompt_mask`;
`DataCollatorMultimodalSeq2Seq` calls `prepare_decoder_input_ids_from_labels` unconditionally;
`MultiModalEmbedderModel.forward` accepts the two keys because `remove_unused_columns=False`
sends every batch key to the model; `MultiLingualSeq2SeqTrainer.prediction_step` pops them and
dispatches to an empty-prompt path (`generate()` with no `decoder_input_ids`), a batched path
for a uniform prompt with prefix `[bos, prompt…]`, or a per-sample path for variable-length
prompts. `bos` is `model.config.decoder_start_token_id` (falling back to the generation
config), so the prefix is right for every backbone: `</s>` for M2M100 and BART, `<pad>` for
T5, the language id for mBART-cc25. Training and inference now start the decoder the same way.

## Consequences

- Models trained before aa9aca7 saw a different decoder prefix in training (position 0 =
  `decoder_start_token_id`) than the one used at inference (position 0 = language token).
  Their reported metrics were computed with a prefix the model was never trained on, so
  results obtained with those checkpoints are not comparable with post-fix runs without
  re-evaluation. Re-run `multimodalhugs-generate` on the old checkpoint with the fixed code
  (the weights are unaffected; only the inference prefix changes) before comparing, and expect
  the numbers to move in either direction.
- Training with `label_smoothing_factor > 0` together with `eval_on_start` or
  `dataloader_num_workers > 0` no longer crashes.
- Breaking for saved processors and `slots:`-format YAML: the slot keys are stored in the
  saved processor, so every processor built before the fix must be rebuilt with
  `multimodalhugs-setup --do_processor`; `slots:` configs must first rename
  `output_data_key: decoder_input_ids` to `decoder_prompt_ids` and
  `output_mask_key: decoder_attention_mask` to `decoder_prompt_mask`. The `pipeline:`
  shorthand needs no YAML change. The TSV columns are unchanged.
- `MultiModalEmbedderModel.forward` carries two parameters it never reads; they go away if
  the prompt is ever moved out of the batch.
- Tests and `tests/TESTS.md` were updated in 78346fe.
