# CLAUDE.md — MultiModalHugs

## Project Overview

MultiModalHugs is a modular framework built on Hugging Face for training, evaluating, and deploying multimodal AI models. Primary focus: sign language translation and multimodal machine translation. Supports input modalities: pose sequences, video, images, SignWriting, precomputed features, and text.

**Version:** 0.4.0
**License:** MIT
**Python package:** `multimodalhugs`

## Build & Run

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_config/
pytest tests/test_model_only/
pytest tests/e2e_overfitting/

# CLI entry points
multimodalhugs-setup   # or mmhugs-setup   — initialize datasets, processors, models
multimodalhugs-train   # or mmhugs-train   — train a model
multimodalhugs-generate # or mmhugs-generate — evaluate and generate predictions
```

## Key Dependencies

- `transformers <= 4.44.2`, `torch < 2.6`, `datasets`, `accelerate`
- `pose-format >= 0.10.1`, `opencv-python`, `av >= 10.0.0`
- `sacrebleu`, `evaluate`, `jiwer` (metrics)
- `omegaconf` (YAML config), `sentencepiece`, `sacremoses` (tokenization)
- Dev: `pytest`, `black`, `isort`, `pylint`

## Project Structure

```
multimodalhugs/
├── data/                    # Data handling
│   ├── datasets/            # HF GeneratorBasedBuilder subclasses (pose2text, video2text, etc.)
│   ├── dataset_configs/     # Data config classes
│   └── datacollators/       # Batching logic (MultimodalDataCollator)
├── models/                  # Model architectures
│   └── multimodal_embedder/ # Main model: FeatureExtractor + MultimodalMapper + Backbone
├── modules/                 # Reusable components (adapters, mappers, embeddings)
├── processors/              # Input processors per modality
├── tasks/translation/       # Training + generation dispatchers, config dataclasses
├── training_setup/          # Per-modality setup (dataset building, processor init, model creation)
├── multimodalhugs_cli/      # CLI entry points (train, generate, setup)
├── utils/                   # Registry, tokenizer utils, general helpers
└── multilingual_seq2seq_trainer.py  # Custom Seq2SeqTrainer
```

## Architecture

**Three-component model** (`MultiModalEmbedderModel`):
1. **FeatureExtractor** — wraps pretrained vision/audio models (CLIP, ViT, etc.)
2. **MultimodalMapper** — maps features to embedding space (linear/adapter/cnn_adapter)
3. **Backbone** — seq2seq model (M2M-100, mBART) for text generation

**Data flow:** Raw data (TSV + media) → Dataset → Processor → DataCollator → Model → Trainer → Metrics

**Configuration:** 3-tier system — YAML files + CLI args + dataclass defaults, merged via `merge_config_and_command_args()`.

## Key Patterns

- **Registry pattern:** `@register_model`, `@register_dataset` decorators for dynamic loading
- **Auto-registration:** `models/__init__.py` registers with HF's `AutoConfig`/`AutoModelForSeq2SeqLM`
- **Composition:** Model composes feature extractor, mapper, and backbone
- **HF-native:** Extends `PreTrainedModel`, `Seq2SeqTrainer`, `GeneratorBasedBuilder`
- **Slot-based processors:** `MultimodalMetaProcessor` composes `ModalityProcessor` instances via `ProcessorSlot` objects; legacy task-specific wrappers live in `processors/legacy/`

## Dataset Format

- TSV metadata files with columns: `signal`, `signal_start`, `signal_end`, `encoder_prompt`, `decoder_prompt`, `output`
- Separate TSVs for train/val/test splits

## Test Patterns

- **pytest** with parametrization (`@pytest.mark.parametrize`)
- **Fixture-based:** `model_setup` fixture for config variations, scoped per function
- **Seed control:** torch, numpy, random, CUDA seeds + `cudnn.deterministic=True`
- **Overfitting tests:** Train for N epochs, assert loss < threshold and WER <= threshold
- **Config tests:** Validate max_length and backbone config behavior

## Supported Modalities

| Modality     | Dataset Class              | ModalityProcessor                    | Legacy wrapper (processors/legacy/)      |
|-------------|---------------------------|--------------------------------------|------------------------------------------|
| Pose        | Pose2TextDataset          | PoseModalityProcessor                | Pose2TextTranslationProcessor            |
| Video       | Video2TextDataset         | VideoModalityProcessor               | Video2TextTranslationProcessor           |
| Image       | BilingualImage2TextDataset| ImageModalityProcessor               | Image2TextTranslationProcessor           |
| SignWriting | SignWritingDataset        | SignwritingModalityProcessor         | SignwritingProcessor                     |
| Features    | Features2TextDataset      | FeaturesModalityProcessor            | Features2TextTranslationProcessor        |
| Text        | BilingualText2TextDataset | TextModalityProcessor                | Text2TextTranslationProcessor            |

## Style & Conventions

- Use `black` for formatting, `isort` for imports
- Follow existing HF conventions for model/config/processor classes
- YAML configs in `examples/` directories serve as templates

## Development Environment

**Python:** 3.11+ recommended, 3.8+ supported.

```bash
# if necessary, install Python 3.11
pyenv install 3.11

# Create a venv and install with dev dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Skip e2e tests if the full suite takes too long (they download models from HuggingFace)
pytest tests/ --ignore=tests/e2e_overfitting -v
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
because they contain absolute paths. After cloning, regenerate them once:

```bash
python tests/assets/generate_assets.py
```

Text and image TSVs are committed directly and do not need regeneration.
