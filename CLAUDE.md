# CLAUDE.md — MultiModalHugs

## Rules

The binding contract for design, coding, testing, documentation, commits and refactoring is
`AGENTS.md` at the repository root. It is imported here so that it loads with this file:

@AGENTS.md

Read it before any technical work. This file holds only facts about the project (layout,
commands, dependencies, modalities); when the two disagree, `AGENTS.md` wins for rules and the
code wins for facts.

## Project Overview

MultiModalHugs is a modular framework built on Hugging Face for training, evaluating, and deploying
multimodal AI models. Primary focus: sign language translation and multimodal machine translation.
Supports input modalities: pose sequences, video, images, SignWriting, precomputed features, and text.

**Version:** 0.5.4
**License:** MIT
**Python package:** `multimodalhugs`

## Build & Run

```bash
# Install (editable, every modality extra plus dev tools)
pip install -e ".[full,dev]"

# Fast suite (everything except the end-to-end runs, which download models)
pytest --ignore=tests/e2e_overfitting

# Full suite, required before a PR is marked ready for review
pytest

# Format and lint the files you changed (configuration in pyproject.toml)
ruff format <files>
ruff check <files>

# CLI entry points
multimodalhugs-setup    # or mmhugs-setup    — initialize datasets, processors, models
multimodalhugs-train    # or mmhugs-train    — train a model
multimodalhugs-generate # or mmhugs-generate — evaluate and generate predictions
```

`pyproject.toml` sets `testpaths = ["tests"]` and `addopts = "-v"`, so `pytest` alone runs the
suite verbosely; `pytest tests/test_data/` and the other tier directories run one tier.

## Key Dependencies

Versions and caps are declared in `pyproject.toml` and are not repeated here.

- Core: `transformers`, `torch`, `datasets`, `accelerate`
- Metrics: `sacrebleu`, `evaluate`, `jiwer`
- Config and tokenization: `omegaconf` (YAML), `sentencepiece`, `sacremoses`
- Optional extras, imported lazily by the code path that needs them: `pose` (`pose-format`),
  `video` (`av`, `torchvision`, `opencv-python`), `signwriting` (`signwriting`), `image`
  (`opencv-python`); `full` installs all of them
- Dev: `pytest`, `ruff`

## Project Structure

```
multimodalhugs/
├── data/                    # Data handling
│   ├── datasets/            # HF GeneratorBasedBuilder subclasses (pose2text, video2text, etc.)
│   ├── dataset_configs/     # Data config classes
│   └── datacollators/       # Batching logic (DataCollatorMultimodalSeq2Seq)
├── models/                  # Model architectures
│   └── multimodal_embedder/ # Main model: FeatureExtractor + MultimodalMapper + Backbone
├── modules/                 # Reusable components (adapters, mappers, embeddings)
├── processors/              # ModalityProcessor implementations, ProcessorSlot, MultimodalMetaProcessor
│   └── legacy/              # Deprecated task-specific processor wrappers
├── tasks/translation/       # Training + generation dispatchers, config dataclasses
├── training_setup/          # General setup (dataset building, processor init, model creation)
├── multimodalhugs_cli/      # CLI entry points (train, generate, setup)
├── utils/                   # Registry, tokenizer utils, general helpers
└── multilingual_seq2seq_trainer.py  # Custom Seq2SeqTrainer
```

## Architecture

**Three-component model** (`MultiModalEmbedderModel`):
1. **FeatureExtractor** — wraps pretrained vision/audio models (CLIP, ViT, etc.)
2. **MultimodalMapper** — maps features to embedding space (linear/adapter/cnn_adapter)
3. **Backbone** — seq2seq model (M2M-100, mBART, T5 family) for text generation

**Data flow:** Raw data (TSV + media) → Dataset → Processor → DataCollator → Model → Trainer → Metrics

**Configuration:** 3-tier system — YAML files + CLI args + dataclass defaults, merged via
`merge_config_and_command_args()` (`tasks/translation/utils.py`). An explicit command-line flag
wins over the YAML value, which wins over the dataclass default.

## Key Patterns

- **HF-native:** extends `PreTrainedModel`, `ProcessorMixin`, `Seq2SeqTrainer`, `GeneratorBasedBuilder`.
- **Composition:** the model composes feature extractor, mapper, and backbone.
- **Auto-registration:** `models/__init__.py` registers model/config classes with HF's
  `AutoConfig`/`AutoModelForSeq2SeqLM`; `processors/__init__.py` registers processors with `AutoProcessor`.
- **Model registry:** `@register_model` (`utils/registry.py`); `build_and_save_model` resolves
  `model.type` through it.
- **Dataset selection:** today the framework resolves `data.dataset_type` through
  `_DATASET_IMPORT_MAP` in `training_setup/general_training_setup.py`; `@register_dataset`
  exists but is not consulted at runtime (used only by `scripts/generate_dataset_configs_docs.py`).
  The target design is a registry per component kind; see `AGENTS.md` §4 and the open issue.
- **Slot-based processors:** `MultimodalMetaProcessor` composes `ModalityProcessor` instances via
  `ProcessorSlot` objects, declared in YAML as `slots:` or the `pipeline:` shorthand
  (`docs/processors/processor_config_formats.md`). User-defined processor classes are supplied to
  `MultimodalMetaProcessor.from_pretrained(..., processor_registry={...})`.

## Dataset Format

- TSV metadata files with columns: `signal`, `signal_start`, `signal_end`, `encoder_prompt`, `decoder_prompt`, `output`
- Separate TSVs for train/val/test splits
- Field semantics: `docs/README.md`, section "Metadata File Requirements"

## Tests

Catalogue of every test and what it guarantees: `tests/TESTS.md`. Policy (approval before any
test change, `TESTS.md` updated in the same commit): `AGENTS.md` §7.

- `tests/test_data/` — fast unit and contract tests, offline
- `tests/test_model_only/` — model construction and tiny training runs with the committed tiny tokenizer, offline
- `tests/e2e_overfitting/` — end-to-end setup → train → generate; downloads a tiny backbone
- Seeds are set per test by an autouse fixture in `tests/test_data/conftest.py`

## Supported Modalities

| Modality     | Dataset Class              | ModalityProcessor                    | Legacy wrapper (processors/legacy/)      |
|-------------|---------------------------|--------------------------------------|------------------------------------------|
| Pose        | Pose2TextDataset          | PoseModalityProcessor                | Pose2TextTranslationProcessor            |
| Video       | Video2TextDataset         | VideoModalityProcessor               | Video2TextTranslationProcessor           |
| Image       | BilingualImage2TextDataset| ImageModalityProcessor               | Image2TextTranslationProcessor           |
| SignWriting | SignWritingDataset        | SignwritingModalityProcessor         | SignwritingProcessor                     |
| Features    | Features2TextDataset      | FeaturesModalityProcessor            | Features2TextTranslationProcessor        |
| Text        | BilingualText2TextDataset | TextModalityProcessor                | Text2TextTranslationProcessor            |

The list grows; the code is authoritative (`training_setup/general_training_setup.py`,
`processors/__init__.py`).

## Style & Conventions

- Formatting and linting: `ruff`, configured in `pyproject.toml`; run it on the files you touch.
- Code, docstring, naming and logging standards: `AGENTS.md` §5.
- Follow existing HF conventions for model/config/processor classes (`AGENTS.md` §3).
- YAML configs in `examples/` directories serve as templates and must stay runnable.

## Development Environment

**Python:** the floor is `requires-python` in `pyproject.toml`.

```bash
# if necessary, install a supported Python
pyenv install 3.11

# Create a venv and install with every extra plus dev tools
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,dev]"

# Run the fast suite
pytest --ignore=tests/e2e_overfitting
```

**Known issue — TensorFlow mutex crash on macOS ARM:**
TF 2.20+ uses protobuf 6.x, which conflicts with PyArrow (built against protobuf 5.x), causing a
`mutex lock failed: Invalid argument` crash during pytest collection. This is triggered because
`transformers` auto-initializes TF at import time when it detects TF is installed.

**Fix:** Use Python 3.11 with `mediapipe<0.10.30`. This combination avoids the crash.
See: https://github.com/tensorflow/tensorflow/issues/98563

## Test Assets

Some tests rely on committed binary assets in `tests/assets/` (pose files, videos, `.npy` files).
The TSV metadata files for path-dependent modalities (video, pose, features) are **not committed**
because they contain absolute paths. `tests/conftest.py` regenerates them automatically at the
start of every pytest session. Run the generator by hand only when you use the assets outside
pytest:

```bash
python tests/assets/generate_assets.py
```

Text and image TSVs are committed directly and do not need regeneration.