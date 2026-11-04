# Handoff — transformers 5 migration

Branch: `update_transformers_compatibility`
Goal: multimodalhugs runs on `transformers>=5.5,<6` with the full suite green, the two
reproduction jobs below land within noise of the master baseline (every difference explained,
none ignored), and the pull request is merged into `master`.

## Done

- The 38 incompatibilities found and resolved during the migration, with the fix for each,
  are in `compat_transformers_5.md`; `git log master..HEAD` is the commit-by-commit view.
- Branch close-out, 2026-11-03:
  - 71d4354 `[docs] Commit the transformers 5 compatibility record`
  - 2d99815 `[chore] Remove verification scripts`
  - fef66ea `[fix] Remove deprecation warnings that name classes not yet available`
  - 2bffcba `[docs] Add decoder_prompt_ids decision note`
  - e3073f9 `[docs] Add CHANGELOG entries for the transformers 5 migration`
  - b16421e `[docs] Rewrite the migration handoff note`
- Follow-ups, 2026-11-04:
  - 122be9b `[docs] Record the prediction_step gen_kwargs override removal` (entry 23 f)
  - 1d4a52d `[docs] Update decoder_input_ids references to decoder_prompt_ids`
  - `[fix] Map legacy decoder_input_ids slot keys to decoder_prompt_ids on load` (the commit
    that carries this line; record entry 39): legacy wrappers and 0.5.x saved processors
    deliver the decoder prompt to generation again.

## Remaining

1. Run the two reproduction jobs on the cluster and record the comparison table here.
   - Job 1, training: same training config and seed as the master baseline run, video
     modality, M2M100 backbone, fp16. Compare validation metric per step, final test
     sacrebleu and chrf, train loss per step, s/it and GPU memory, and a sample of decoded
     outputs. Job 1 may differ legitimately: `compat_transformers_5.md` entry 36
     (`decoder_prompt_ids.md`) changed the decoder prefix used for generation, so every
     validation and test metric is computed with a prefix the master run never used. A
     difference there is explained, not ignored; the train loss per step is not affected by
     entry 36 and should match.
   - Job 2, generation only: the checkpoint `c79ae10_hf_1_video/train/checkpoint-best`
     generated on both versions with the same test set. Compare metrics and decoded outputs.

   | Measure | master | this branch | Explained by |
   |---|---|---|---|
   | Job 1: validation metric per step | | | |
   | Job 1: final test sacrebleu / chrf | | | |
   | Job 1: train loss per step | | | |
   | Job 1: s/it, GPU memory | | | |
   | Job 1: decoded sample | | | |
   | Job 2: test sacrebleu / chrf | | | |
   | Job 2: decoded sample | | | |

2. Open the pull request against `master` (description per `AGENTS.md` §10: motivation,
   summary, how it was tested including the table above, what to review closely).
3. After the merge, release as 0.6.0: minor, because the branch carries breaking changes
   (`CHANGELOG.md`, `[Unreleased]`), per `AGENTS.md` §8. Then delete this note.
4. Not part of this branch: the planned rename of the model family to `ModularMultiModal*`
   and its deprecation path are a separate later effort.

## How to resume

- Environment: conda env `mmhugs`; run everything as `conda run -n mmhugs <command>`.
  Measured on 2026-11-03: Python 3.11.18, torch 2.10.0, transformers 5.22.1.
- Install: `pip install -e ".[full,dev]"`.
- Fast suite: `conda run -n mmhugs pytest --ignore=tests/e2e_overfitting -q`
  → 444 passed in 35 s (2026-11-03, at fef66ea).
- Full suite: `conda run -n mmhugs pytest -q` (adds `tests/e2e_overfitting`, which downloads
  `google/t5-efficient-tiny`) → 447 passed in 317 s (444 fast + 3 e2e) (2026-11-03, at e3073f9).
- Material moved out of the tree on 2026-11-03 to `~/multimodalhugs_local/` on the
  maintainer's machine: `debug_option_a/` (How2Sign video config, TSVs, srun script),
  `reproduce_runs/` (training logs and arguments of the `c79ae10` and `main_branch`
  reproduction runs), `remote_debug_option_a_prompt.md`,
  `remote_debug_torchcodec_gpu_workers.md`, `transformers5_multimodal_opportunities.md`.
- `ruff format` was not run on the two model files touched by fef66ea: it would rewrite
  about 360 unrelated lines, and the tree-wide format is a separate chore commit
  (`CHANGELOG.md`, Development tooling). `ruff check` reports the same 219 pre-existing
  findings before and after.

## Open questions for the maintainer

- Was torchcodec GPU decode (`backend: torchcodec`, `device: cuda`) inside DataLoader
  workers with `worker_start_method: spawn` verified on the cluster? The local note that
  planned that check (`remote_debug_torchcodec_gpu_workers.md`) was moved out of the tree
  on 2026-11-03, and nothing in the repository records its outcome.
- Which run is the master baseline for Job 1? `c79ae10` is the merge of `master` into this
  branch, not a `master` commit, and the moved `reproduce_runs/` holds logs for both a
  `c79ae10` run and a `main_branch` run.
