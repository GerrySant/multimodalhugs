# AGENTS.md — the contract for working on multimodalhugs

This file binds every coding agent (Claude Code, Codex, Cursor, …) and every human contributor.
It says how work is done here and what standard code must meet. It takes precedence over a
request made in conversation: when a request conflicts with it, say so, propose the compliant
alternative and wait for the maintainer's decision. When a rule here is unclear or looks wrong,
propose a change to this file; the one thing it does not allow is silent deviation.

The rules describe the standard new code is held to and the direction every refactor moves in.
Parts of the current code do not meet them. That is a reason to refactor deliberately (§9), not
a licence to add more of the same.

Facts about the project (map, commands, supported modalities, environment) live in `CLAUDE.md`
and `docs/README.md` and are not repeated here.

---

## 1. Where truth and state live

- The code and the tests are the truth. `CLAUDE.md`, `docs/`, docstrings and `CHANGELOG.md`
  orient you; verify every claim that matters against the code before relying on it, because
  documentation here lags the code.
- Open a file before speaking about it, and search before stating that something does not
  exist. Read the supported modalities, models and tasks from the code; they grow.
- When documentation and code disagree, say so (which file, what it says, what the code does),
  treat the code as correct, and fix the documentation in the change at hand.
- Current state is never recorded in this file. It lives in GitHub issues and pull requests, in
  `docs/design_notes/` (one note per decision; one handoff note per long-running effort) and in
  `CHANGELOG.md` under `[Unreleased]`. Read those three at the start of a session before asking
  what to do.
- The catalogue of what every test guarantees is `tests/TESTS.md`. Tooling configuration
  (Python floor, formatter, linter, pytest) is `pyproject.toml`.

---

## 2. Working method

- Restate the goal and the acceptance criteria in one or two sentences before starting.
- Ask one precise question when a wrong guess would be expensive to undo; otherwise decide, say
  which reading you took, and continue. The maintainer decides what the specified behaviour is;
  you decide how to implement it.
- Read the code you will change, its tests and its docs, and search for every call site of
  anything whose signature or behaviour you touch, because the repository is large and only
  part of it is ever in view.
- Before touching the Hugging Face surface, open the nearest upstream analogue
  (`M2M100ForConditionalGeneration`, `Seq2SeqTrainer`, `ProcessorMixin`, …) in the installed
  version (`python -c "import transformers, inspect; …"`) and follow it; memory of older
  versions is not evidence.
- For work that touches more than about three files, any public interface, or any refactor,
  write a short plan (files to touch, order of commits, tests that prove it) and get it approved
  before coding.
- Do what was asked. Note problems you pass by in the final message or in an issue; fix them in
  the same change only when they block it.
- Report only what you ran: paste the test outcome, and say explicitly when something could not
  be run here (GPU, download, time). Label claims as verified (read in the code), inferred
  (follows from what you read) or a guess, so the reader knows which parts to trust.
- Close with what changed, what was verified and what remains. Say what is blocking when you
  cannot finish, and leave no half-applied change in the tree unannounced.

---

## 3. Hugging Face alignment

Adapt multimodalhugs to the Hugging Face API rather than patching around it, because every
shim, every `try/except` around a removed API and every version branch becomes a maintenance
liability at the next upgrade and hides the real divergence.

- Follow the installed upstream pattern, including naming. When HF removes a config attribute,
  remove it from our configs. When HF renames a parameter, rename ours and accept only the new
  name. When HF changes a return type, write a general extraction helper that works for any
  model, not a special case for the model at hand.
- Prefer HF utilities and base classes over our own: `load_image`, `load_video`,
  `GenerationConfig`, `ProcessorMixin` save/load, `PreTrainedModel` weight tying and
  initialisation, `Trainer` hooks, `Auto*` registration.
- Keep our own abstractions (`ModalityProcessor`, `ProcessorSlot`, `MultimodalMetaProcessor`,
  the mappers, the registries) only where HF has no equivalent, and plan a migration when HF
  gains one.
- The test is API alignment, not the absence of crashes: a fix that makes tests pass while
  keeping a pattern HF has abandoned is not done.
- During a dependency major-version upgrade, record each incompatibility (symbol, files
  affected, what changed, the fix) in a compatibility document inside the repository before
  committing its fix, and commit one resolved incompatibility at a time, so that the next
  upgrade can learn from this one.

---

## 4. Architecture

- Stable core, extensible periphery. The core is the small set of contracts and mechanisms every
  modality shares: the processor contracts, the dataset base classes, the collator, the trainer,
  the registries and the config merging. Modalities, models, mappers, metrics and tasks are
  periphery. Adding a feature leaves the core untouched, so that the hundredth feature costs
  what the tenth did; a feature that needs a core change is a design discussion first.
- Layers depend in one direction: `data → processors → models → training → tasks → cli`. A
  lower layer imports nothing from a higher one, `multimodalhugs/` imports nothing from `tests/`
  or `examples/`, and an import cycle is a bug.
- Compose; subclass only HF bases. A model has a feature extractor, a mapper and a backbone.
  Subclass `PreTrainedModel`, `ProcessorMixin`, `Seq2SeqTrainer` and `GeneratorBasedBuilder`,
  because that is what makes us HF-native, and build no inheritance towers on top of them.
- Registries, not conditionals. A dataset, model, processor or metric is added by subclassing
  its skeleton base class (each shipped with a worked example), registering the subclass, and
  letting the framework resolve it through that registry; a conditional chain on a type name
  means every new kind edits the core.
- Contracts are small and explicit: one to four mandatory methods, the rest optional with sane
  defaults, so that a new implementation is cheap to write and hard to leave half done.
- Configuration is data: dataclasses, `PretrainedConfig` subclasses or YAML, validated at
  construction (`__post_init__`), serialisable, and sufficient to rebuild the object with
  `from_pretrained` and no other code. Defaults cover the common case in one YAML block.
- Abstract on the third case. Two concrete cases may stay duplicated; with three you know what
  varies. Add parameters, hooks and base classes for needs that exist today.
- Backward compatibility is a feature. Saved checkpoints, saved processors and user YAML from
  the previous minor version keep loading, because users have experiments tied to them. The
  dataset TSV columns (`signal`, `signal_start`, `signal_end`, `encoder_prompt`,
  `decoder_prompt`, `output`) are public API. Anything that cannot be kept goes through the
  deprecation path (§8).

---

## 5. Code standards

Explicit over implicit

- Pass arguments by name. Forward `**kwargs` through at most one layer, and document which keys
  go where; tunnelled kwargs hide the real signature.
- Keep state local: no hidden globals, no behaviour driven by undocumented environment
  variables, no mutation of module-level singletons at import time. Registration decorators are
  the one sanctioned import-time effect; importing `multimodalhugs` downloads nothing,
  initialises no CUDA and touches no filesystem.
- Fail loudly. When a requested backend, device or feature is unavailable, raise or warn; a
  silent fallback turns a configuration mistake into a wrong experiment.

Errors

- Validate at construction time, and say what is wrong, what value was received and what to do:
  `ValueError(f"signal_start_end_unit must be one of {list(SignalUnit)}, got {unit!r}")`.
- Catch the narrowest exception you can actually handle. `assert` guards internal invariants
  only, never user input.

Immutability

- Configs and data records are not mutated after creation; derive values in `__post_init__` or
  in a function that returns a new object. A merging helper returns a new object, and callers
  use the return value.

Types and names

- Full type hints on every public function and method, return type included. `Optional[X]`
  only when `None` means something, and the docstring says what.
- Names carry units and semantics (`max_frames`, `signal_start_ms`, `feat_dim`); booleans read
  as predicates (`reduce_holistic_poses`, `is_label`); closed sets of values are enums
  (`SignalUnit`, `TextRole`); parameters beyond the first one or two are keyword-only.

Tensors and numerics

- Document the shape of every tensor argument and return value (`[B, T, C, H, W]`) in the
  docstring, and check rank at the boundary when it is cheap.
- Take device and dtype from the inputs or from an explicit parameter; use `.to(device)`, not
  `.cuda()`; the backbone and `TrainingArguments` own fp16/bf16.
- Keep numerical logic (normalisation, masking, metrics, losses) in pure functions, tensors in
  and tensors out, so that it can be tested without a model.
- Randomness goes through an explicit `torch.Generator` or the seed the trainer sets; a
  processor is deterministic given a seed.

Dependencies

- Core dependencies stay minimal. Modality-specific and integration packages are optional
  extras, imported lazily inside the code path that needs them, with an `ImportError` naming
  the extra to install; the import-guard tests in `tests/test_data/` stay green.
- A version cap is temporary and carries, inline, the issue number and the condition for its
  removal; a cap without an exit condition is a bug. Pin ranges, not exact versions.
- The Python floor declared in `pyproject.toml` (`requires-python`) is binding: use no syntax or
  standard-library feature above it, and keep no backport for versions below it.

Logging

- `logging.getLogger(__name__)` inside the package, never `print`. `warnings.warn(…,
  FutureWarning)` for deprecations, because `DeprecationWarning` is hidden from library users by
  default; `logger.warning` for runtime conditions, emitted once rather than once per element.

Style

- Run `ruff format` and `ruff check` as configured in `pyproject.toml` on the files you touch,
  and only on those, so that a diff shows one change.
- Google-style docstrings that say why and under which constraints, not a restatement of the
  signature; a public class carries a usage example. Comments record decisions and trade-offs;
  delete a comment that describes code that no longer exists.

---

## 6. Research correctness

Clean code that silently alters results is a research bug, the most expensive kind this project
has, so the behaviours below are treated as behaviour, never as plumbing.

- Tokenisation and special tokens, decoder prefixes and start tokens, normalisation, frame
  sampling and strides, temporal clipping, seeding, mixed precision, and what is decoded and
  stripped before a metric is computed. A change to any of them states its consequence for
  experiments in the pull request and ships with a test that pins the new behaviour.
- Keep training and inference consistent: the processor that built the training batch is the
  processor that builds the inference batch, with the same configuration.
- A numerical or behavioural regression is a finding to explain, never a test to update. A
  result that looks too good (a large jump from a small change) is checked against the decoded
  outputs before it is reported.
- When an engineering choice has a modelling consequence (what the backbone sees, how a
  sequence is padded or truncated, which tokens the loss ignores), say so explicitly.

---

## 7. Tests

Tests are the specification, and only the maintainer changes the specification.

- Before adding, modifying, deleting or skipping any test, explain what will change and why, and
  wait for explicit approval. After an approved change, update `tests/TESTS.md` in the same
  commit, so that the catalogue never drifts from the suite.
- When a test fails, the explanation is "the code is wrong" or "the test is wrong"; say which
  and why. An assertion is never weakened to go green, and `skip`/`xfail` is never a way to
  finish a task.
- Tiers (the catalogue is `tests/TESTS.md`): `tests/test_data/` holds fast unit and contract
  tests, no GPU, no network, and is where most new tests go; `tests/test_model_only/` holds
  construction, save/load and tiny forward passes with the committed tiny tokenizer, offline;
  `tests/e2e_overfitting/` holds small end-to-end runs, the only tier that may download, kept
  few because it is slow; `tests/assets/` holds small committed fixtures and goldens, with
  path-dependent TSVs generated by `tests/conftest.py` rather than committed.
- A good test checks one behaviour and names it (`test_frames_unit_respects_start_end`); it is
  deterministic (seeds set, no wall-clock, no network outside e2e); a bug fix comes with a
  regression test whose docstring cites the issue.
- A new processor, dataset or model joins the shared contract tests (output keys and shapes,
  `save_pretrained` → `from_pretrained` round trip, determinism under a fixed seed, empty and
  edge inputs) rather than copying a bespoke suite.
- Golden files in `tests/assets/golden/` protect numerical output. Regenerate them only with
  approval, stating what changed numerically and why.

---

## 8. Documentation, changelog, deprecation, versioning

- A change to a public class, parameter, YAML key or CLI flag updates the matching page in
  `docs/` in the same pull request, and a new page is added to `docs/README.md`, which is
  maintained by hand. Fix a stale docstring, page or comment you pass through, or flag it when
  it is out of scope.
- A design decision gets a short note in `docs/design_notes/` (context, decision, consequences,
  status; one note per decision), and a long-running effort keeps a handoff note there that is
  updated at the end of every session; both formats are in that directory's `README.md`.
  Compatibility records from a dependency upgrade live in the repository, never in an ignored
  or local file.
- Every user-visible change gets an entry in `CHANGELOG.md` under `[Unreleased]`, in one of
  Added, Changed, Deprecated, Removed or Fixed, in the same pull request. A breaking change is
  marked Breaking and says what the user must do.
- A public API is removed only through a deprecation path: (1) emit `FutureWarning` naming the
  replacement and the removal version; (2) keep the old path working for at least one minor
  version; (3) remove it in the stated version with a CHANGELOG entry.
- Semantic versioning on `0.x`: patch for fixes, minor for features and for breaking changes
  that followed the deprecation path. The version lives in `pyproject.toml` and `CLAUDE.md`;
  only a `[release]` commit bumps both and turns `[Unreleased]` into the version section.
- Keep `CLAUDE.md` and this file accurate; a stale instruction is worse than none.

---

## 9. Refactoring

- A refactor changes no behaviour, and a behaviour change does not refactor, so that every diff
  can be reviewed for one thing.
- A small refactor (one module, no interface change) is allowed as part of a fix or feature only
  when that change needs it, and it goes in its own commit before the behavioural one, with
  tests green before and after.
- A large refactor (interfaces, layering, responsibilities) starts with a design note (problem,
  options, chosen design, migration plan, what stays compatible) approved before code, and
  proceeds incrementally: new path added, call sites migrated one by one with tests, old path
  deprecated, old path removed in a later version; each step is a separate pull request that
  leaves `master` releasable.
- Propose (do not start) a refactor when you find an import cycle, a feature that had to touch
  the core, a conditional chain on a type name, `**kwargs` tunnelled through layers, a shim for
  an API HF has removed, or a class that cannot be tested without a GPU or a download.

---

## 10. Commits and pull requests

- Title: `[tag] Imperative summary`, with the tag one of `fix`, `feat`, `refactor`, `test`,
  `docs`, `arch`, `chore`, `ci`, `release`. Example:
  `[fix] Respect signal_start_end_unit in Video2TextDataset duration filter`.
- Body: which files, what changed, why, and how it was verified, so that a reviewer understands
  the change without opening the diff.
- One logical change per commit; during a migration, one commit per resolved incompatibility.
- Commits carry no `Co-Authored-By` or tool-attribution trailer; attribution is the pull
  request's job.
- The tree receives no scratch or verification scripts, local notes, absolute paths,
  credentials or large binaries. A script that was useful during development is deleted before
  the pull request is ready, or promoted into a real test with approval.
- One pull request per issue, referencing it (`Closes #N`). The description gives the
  motivation, a summary of the changes, how it was tested, what reviewers should look at
  closely, and the follow-ups deliberately left out. A pull request is ready for review when
  the full suite including `tests/e2e_overfitting` is green locally, `docs/`, `CHANGELOG.md` and
  `tests/TESTS.md` are updated, the diff holds no work-in-progress artefacts, and the branch is
  rebased on `master`.
- `master` is releasable at all times.

---

## 11. Definition of done

Confirm every line before reporting a task complete:

- [ ] The change does what was asked, and nothing else (§2).
- [ ] HF alignment respected: no shim, no `try/except` around a removed API (§3).
- [ ] Layering preserved, no new import cycle, core untouched or discussed first (§4).
- [ ] Public interfaces unchanged, or a deprecation path in place (§8).
- [ ] Type hints, shapes and units in docstrings, explicit errors (§5).
- [ ] Optional dependencies imported lazily with a named `ImportError` (§5).
- [ ] Research-relevant behaviour unchanged, or its consequence stated and tested (§6).
- [ ] Tests added or changed only with approval; `tests/TESTS.md` updated (§7).
- [ ] `pytest --ignore=tests/e2e_overfitting` run and its result reported (§2).
- [ ] `docs/` and `CHANGELOG.md [Unreleased]` updated for every public change (§8).
- [ ] Commits follow §10; `ruff format` and `ruff check` clean on the touched files (§5).
- [ ] The final message states what changed, what was verified and what remains (§2).