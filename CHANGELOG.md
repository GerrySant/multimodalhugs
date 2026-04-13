# Changelog

All notable changes to the project are documented in this file.

Version numbers are of the form `1.0.0`.

Each version section may have subsections for: _Added_, _Changed_, _Removed_, _Deprecated_, and _Fixed_.

## [0.5.3]

### Changed

- **`multimodalhugs-generate` now supports multiple metrics via `--metric_name`.**
  `metric_name` accepts a comma-separated list (e.g. `sacrebleu,chrf`). Each metric is evaluated independently and stored under its own key in the results. Single-metric usage is unchanged.

---

## [0.5.2]

### Added

- **`signal_start_end_unit` parameter on `PoseModalityProcessor`, `VideoModalityProcessor`, `Pose2TextDataConfig`, and `Video2TextDataConfig`.**
  Processors and dataset configs now accept `signal_start_end_unit` (default `SignalUnit.MILLISECONDS`, preserving the existing behaviour). Setting it to `SignalUnit.FRAMES` tells the processor to interpret `signal_start` / `signal_end` as frame indices rather than milliseconds.

  - **`PoseModalityProcessor`** — passes `start_frame` / `end_frame` directly to `Pose.read`, which uses a seek-capable `BytesIOReader` internally. Normalisation sees only the requested window, consistent with the milliseconds path. Raises `ValueError` at construction time for unknown unit values.

  - **`VideoModalityProcessor`** — for the OpenCV path (`custom_preprocessor_path` set), uses `CAP_PROP_POS_FRAMES` for seeking and position checking. For the torchvision path, reads the full video and slices by index (torchvision does not support frame-index seeking natively). Raises `ValueError` at construction time for unknown unit values.

  - **`Pose2TextDataConfig` / `Video2TextDataConfig`** — expose the same parameter; the dataset duration-filtering logic (`mapping_function`) now respects the chosen unit when computing clip duration for `max_frames` / `min_frames` filtering.

  - **Legacy wrappers** (`Pose2TextTranslationProcessor`, `Video2TextTranslationProcessor`) accept and forward the new parameter to their underlying modality processor unchanged.

  - The zero/zero convention (`signal_start=0, signal_end=0` → full file) is preserved for both units.

- **`SignalUnit` enum** (`multimodalhugs.processors.SignalUnit`).
  A `StrEnum`-based enum (Python 3.8+ compatible) that defines the valid values for `signal_start_end_unit`:

  ```python
  from multimodalhugs.processors import SignalUnit

  processor = PoseModalityProcessor(signal_start_end_unit=SignalUnit.FRAMES)
  ```

  Instances compare equal to their string values (`SignalUnit.MILLISECONDS == "milliseconds"`) and serialise as plain strings, so existing YAML configs and `save_pretrained`/`from_pretrained` round-trips are unaffected.

---

## [0.5.1]

### Fixed

- **Partial installation now works as expected.** Previously, importing any part of `multimodalhugs` required all modality-specific dependencies (`pose-format`, `opencv-python`, `signwriting`, `av`, `torchvision`) to be installed, even if the user only intended to use a single modality (e.g. text-to-text). The root causes and their fixes are described below.

- **`multimodalhugs/data/__init__.py`**: Dataset classes (`Pose2TextDataset`, `Video2TextDataset`, etc.) were imported eagerly at package load time, pulling in their optional dependencies immediately. Replaced with PEP 562 `__getattr__` lazy loading — each dataset module is only imported when the class is explicitly accessed.

- **`multimodalhugs/training_setup/general_training_setup.py`**: `_build_dataset_map()` imported all six dataset modules upfront, so running `mmhugs-setup` for any modality (e.g. `features2text`) would also import `pose2text.py`, triggering a `NameError` on the `-> Pose` return annotation when `pose-format` was not installed. Replaced with a `_DATASET_IMPORT_MAP` string table and `_load_dataset_classes(dataset_type)` that only imports the one module actually needed. `_build_dataset_map()` is retained as a public helper (now lazy) for tests and tooling.

- **`multimodalhugs/data/utils.py`**: `from torchvision.transforms import ...` was imported unconditionally at module level, causing `ModuleNotFoundError` for users in pose-only or text-only environments. Wrapped in `try/except ImportError` with a `_TORCHVISION_AVAILABLE` flag.

- **`multimodalhugs/data/datasets/features2text.py`**: Removed a dead `from pose_format import Pose` import that was never used in the file but caused `ModuleNotFoundError` in environments without `pose-format`.

- **`multimodalhugs/__init__.py`**: `from .tasks import *` transitively forced `av` to be imported at package load time via `evaluate → transformers.pipelines.video_classification`. Removed.

- **`multimodalhugs/tasks/translation/translation_training.py`** and **`translation_generate.py`**: `import evaluate` was at module top-level, which transitively imported `transformers.pipelines.video_classification → av`. Moved inside the function body, just before `evaluate.load()` is called. In `translation_training.py` the import is additionally conditional on `metric_name` being set, so environments without a metric configured avoid the `av` dependency entirely.

### Changed

- **`pyproject.toml`**: Modality-specific dependencies moved from core `dependencies` to optional extras. Users can now install only what they need:
  - `pip install "multimodalhugs[full]"` — all modalities (equivalent to the previous default)
  - `pip install "multimodalhugs[pose]"` — pose sequences (`pose-format`)
  - `pip install "multimodalhugs[video]"` — video (`av`, `torchvision`, `opencv-python`)
  - `pip install "multimodalhugs[signwriting]"` — SignWriting (`signwriting`)
  - `pip install "multimodalhugs[image]"` — images (`opencv-python`)
  - Multiple extras can be combined: `pip install "multimodalhugs[pose,video]"`

- **`multimodalhugs/__init__.py`**: Removed `from .tasks import *` and `from .multimodalhugs_cli import *`. The CLI entry points (`mmhugs-train`, `mmhugs-generate`, `mmhugs-setup`) are unaffected — they are declared as `console_scripts` in `pyproject.toml` and call their target functions directly. The training entry points remain accessible via explicit import: `from multimodalhugs.tasks import translation_training_main`.

- **Modality processors** (`PoseModalityProcessor`, `VideoModalityProcessor`, `SignwritingModalityProcessor`, `ImageModalityProcessor`): Optional dependency imports are now wrapped in `try/except ImportError`. A clear `ImportError` with a `pip install` hint is raised at instantiation time if the required package is absent, rather than at module import time.

- **Modality datasets** (`Pose2TextDataset`, `SignWritingDataset`, `Video2TextDataset`): Same lazy-import pattern applied — optional dependency imports wrapped in `try/except`, with a descriptive `ImportError` raised in `__init__` if the dependency is missing.

### Breaking Changes

- **`from multimodalhugs import Pose2TextDataset`** (and other dataset builder classes) no longer works. Dataset builders are internal construction details, not user-facing API — no example, doc, or internal caller relied on the top-level shorthand. Use the fully-qualified form instead: `from multimodalhugs.data import Pose2TextDataset`.

- **`from multimodalhugs.data import *` no longer includes dataset builder classes.** Because the six dataset classes (`Pose2TextDataset`, `Video2TextDataset`, `SignWritingDataset`, `BilingualText2TextDataset`, `BilingualImage2TextDataset`, `Features2TextDataset`) are now lazy-loaded via `__getattr__`, they are excluded from `__all__` and therefore silently absent from a wildcard import. Access them by name: `from multimodalhugs.data import Pose2TextDataset`.

- **`from multimodalhugs import translation_training_main`** (and other `tasks` re-exports) now raises `ImportError`. Removing `from .tasks import *` from `multimodalhugs/__init__.py` broke direct top-level access to task entry points. Use `from multimodalhugs.tasks.translation.translation_training import main` instead.

---

## [0.5.0]

This release introduces a complete redesign of the processor layer, replacing the six monolithic task-specific processors with a modular, composable architecture. The new design separates modality-specific preprocessing from task structure, enables declarative YAML configuration, and provides a unified CLI entry point for all modalities. Full backward compatibility is maintained — existing code, configs, and processor checkpoints continue to work unchanged.

### Added

#### Modality-First Processor Architecture

A new three-layer processor system replaces the previous monolithic task-specific processors:

- **`ModalityProcessor`** (`processors/modality_processor.py`) — Abstract base class for all modality-specific processors. Defines a two-stage interface:
  - `process_sample(values, **kwargs) → Any` — called at dataset-transform time, per item. Handles file I/O (loading `.pose`, `.npy`, video, image, or SignWriting files) and returns a single tensor. The default implementation is a no-op (correct for text).
  - `process_batch(samples, **kwargs) → ProcessBatchOutput` — called at collation time on a list of already-processed samples. Handles padding and mask creation across a batch.
  - A default `__repr__` that displays all JSON-serializable constructor arguments, omitting tokenizers and callables, for clean logging and debugging.

- **`ProcessBatchOutput`** (`processors/modality_processor.py`) — A `NamedTuple(data, mask)` returned by every `process_batch` implementation. `data` and `mask` are both `Optional[torch.Tensor]`. `mask=None` means "no mask to add" (not "all positions valid"); callers must check before inserting into the output batch.

- **`ProcessorSlot`** (`processors/meta_processor.py`) — A `@dataclass` that binds a `ModalityProcessor` instance to its role in the pipeline:
  - `processor` — the modality-specific processor instance.
  - `output_data_key` — key under which the processed data tensor is stored in the model batch (e.g., `"input_frames"`, `"encoder_prompt"`, `"labels"`).
  - `output_mask_key` — key for the attention mask tensor; `None` if no mask is needed.
  - `column_map` — mapping from dataset TSV column names to processor parameter names. The first key is the _primary field_: its value is replaced with a tensor by the dataset transform. Subsequent keys are passed to `process_sample` as keyword arguments but are not written back to the dataset.
  - `is_label` — marks this slot as producing the loss target (`labels`).

- **`MultimodalMetaProcessor`** (`processors/meta_processor.py`) — The central orchestrator. Extends HuggingFace's `ProcessorMixin` and composes a flat, ordered list of `ProcessorSlot` instances into a full pipeline:
  - **Constructor validation**: raises `ValueError` if the slot list is empty, if any two slots share the same `output_data_key`, or if any two slots share the same non-`None` `output_mask_key`.
  - **Tokenizer auto-derivation**: if no tokenizer is provided explicitly, derives it from the first slot whose processor holds one.
  - **`_transform_get_items_output`**: registered with `dataset.with_transform()`. Iterates slots, calls `process_sample` for each slot's primary field, and replaces string values with tensors in-place. Slots whose primary field is absent from the dataset emit a `logger.warning` and are skipped.
  - **`__call__(batch, batch_dict=None)`**: called by the DataCollator. Iterates slots in declaration order, calls `process_batch` for each, and stores the result under `output_data_key` / `output_mask_key`. Slots whose `output_data_key` is already present in the result are skipped (allows external overrides). Returns a HuggingFace `BatchFeature`.
  - **`save_pretrained` / `from_pretrained`**: full serialization round-trip. Saves a `processor_config.json` containing the ordered slot list (class name, constructor kwargs, output keys, column map, `is_label` flag). The tokenizer is saved alongside as a directory. On load, processor classes are looked up by name in `multimodalhugs.processors`; an optional `processor_registry: Dict[str, type]` argument allows user-defined subclasses to be resolved without modifying the library.
  - **`__repr__`**: custom representation showing the slot key→processor mapping and the tokenizer type.

#### Six Concrete ModalityProcessors

One `ModalityProcessor` subclass per supported modality, each living in its own file under `processors/`:

- **`PoseModalityProcessor`** — loads `.pose` files using `pose_format`. Applies `hide_legs`, optional holistic reduction (`reduce_holistic_poses`), normalization, and optional temporal downsampling (`skip_frames_stride`). Accepts a single file path, a pre-loaded tensor, or a dict containing `signal`, `signal_start`, and `signal_end` (temporal clipping in milliseconds).

- **`VideoModalityProcessor`** — reads video files via OpenCV (when a `custom_preprocessor_path` is given) or via `torchvision.io.read_video`. Applies per-frame preprocessing using a HuggingFace `AutoProcessor` (e.g., a CLIP feature extractor), optional temporal downsampling, and an optional channel-height-width merge (`join_chw`). Supports LRU-cached loading (`use_cache`) with cache size automatically derived from available system memory (or SLURM allocation).

- **`FeaturesModalityProcessor`** — loads precomputed feature sequences from `.npy` files. Handles optional temporal axis permutation (`temporal_dimension_position`) and frame downsampling. Supports LRU-cached loading.

- **`ImageModalityProcessor`** — loads images from disk (`.npy`, `.jpg`, `.png`, `.bmp`, `.tiff`) or renders plain text as an image using a font file (`font_path`). Applies optional normalization (`normalize_image`, `mean`, `std`). Validates that the number of normalization channels matches the image's actual channel count. Accepts PyArrow scalars transparently.

- **`SignwritingModalityProcessor`** — converts FSW (Formal SignWriting) ASCII strings to image sequences. Requires a `custom_preprocessor_path` (raises `ValueError` immediately if `None`). Splits the FSW string into individual sign symbols, renders each via `signwriting_to_image`, centres on a white background, optionally inverts colours (`invert_frame=True` by default), and applies a HuggingFace `AutoProcessor` (e.g., CLIP) per frame. Returns a `[N_signs, C, H, W]` tensor.

- **`TextModalityProcessor`** — tokenizes text for two distinct roles selected by a `TextRole` enum:
  - `TextRole.INPUT`: tokenizes a list of strings; returns `(token_ids [B, L], attention_mask [B, L])`.
  - `TextRole.TARGET`: concatenates `target_prefix` + `target` + EOS per sample and pads with −100 (HuggingFace's `CrossEntropyLoss` standard `ignore_index`); returns `(labels [B, L], None)`. Validates that required keys (`target_prefix`, `target`) are present in each sample, providing an actionable message pointing to `column_map` if not.
  - Accepts a pre-built tokenizer object or a `tokenizer_path` / HF Hub ID. When `new_vocabulary` is provided, the vocabulary is extended with `add_special_tokens` and the added tokens are exposed as `self.new_tokens`; the unextended tokenizer is kept as `self.pretrained_tokenizer` for downstream model-construction use.

#### Declarative Processor Configuration

Two new YAML processor config formats enable processor construction without writing Python:

**Pipeline shorthand** — compact form for standard modality pipelines:

```yaml
processor:
  pipeline: video2text           # one of: pose2text, video2text, features2text,
                                 #         image2text, signwriting2text, text2text
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: path/to/vocab.txt   # optional
  modality_kwargs:                    # optional, passed to the modality processor
    skip_frames_stride: 2
    custom_preprocessor_path: openai/clip-vit-base-patch32
```

**Full slots format** — explicit declaration of every slot in the pipeline:

```yaml
processor:
  slots:
    - processor_class: VideoModalityProcessor
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        skip_frames_stride: 2
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map: {signal: signal, signal_start: signal_start, signal_end: signal_end}
    - processor_class: TextModalityProcessor
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        role: target
      output_data_key: labels
      column_map: {decoder_prompt: target_prefix, output: target}
      is_label: true
    # ... additional slots
```

Both formats are handled by two new utility functions in `training_setup/setup_utils.py`:

- **`expand_pipeline_shorthand(processor_cfg)`** — normalises the compact `pipeline:` format to the full `slots:` format. Pure transformation with no processor instantiation.
- **`build_processor_from_config(processor_cfg)`** — instantiates a `MultimodalMetaProcessor` from a slots config. Includes a tokenizer cache (keyed by `tokenizer_path`) that eliminates redundant `AutoTokenizer.from_pretrained` calls when multiple text slots share the same base tokenizer. Emits a `logger.warning` when two slots share `tokenizer_path` but differ in `new_vocabulary` (their tokenizers will have different vocabulary sizes, which is likely incompatible with a single model embedding matrix).
- **`extract_tokenizer_info_from_processor_config(processor_cfg)`** — extracts `(tokenizer_path, new_vocabulary)` from any of the three config formats (shorthand, full slots, or legacy flat) without instantiating any processor. Used when `do_model=True` but `do_processor=False`.

#### Unified Setup CLI

- **`general_training_setup.py`** — new modality-agnostic setup module. A single `main()` function handles all six modalities by reading `data.dataset_type` from the YAML config, selecting the appropriate dataset class, and calling `build_processor_from_config` for the processor. Replaces the need for separate `pose2text_training_setup.py`, `video2text_training_setup.py`, etc. entry points.
- **`multimodalhugs-setup` CLI** now routes to `general_training_setup.main` when `--modality` is omitted, enabling a single command to set up any pipeline regardless of modality.

#### AutoProcessor Registration

- All modality processors and the `MultimodalMetaProcessor` are registered with HuggingFace's `AutoProcessor` in `processors/__init__.py` (via `register_for_auto_class` + `AutoProcessor.register`). This means `AutoProcessor.from_pretrained(saved_processor_path)` works as soon as `multimodalhugs.processors` is imported, regardless of the context (training, inference, or evaluation scripts).

#### New Tests

- **`tests/test_data/test_meta_processor.py`** — comprehensive test suite for `ProcessorSlot` and `MultimodalMetaProcessor`: slot dataclass behaviour, multi-modal pipelines (pose→text, text→text, video+pose dual-encoder), backward-compatibility key parity with legacy processors, DataCollator integration, construction validation (empty slots, duplicate keys), save/load round-trips, `processor_registry` kwarg, missing-column warning, and tokenizer cache scenarios.
- **`tests/test_data/test_modality_processors.py`** — unit tests for all six `ModalityProcessor` subclasses: tensor passthrough, file loading, batch padding, mask shapes, and `ProcessBatchOutput` structure.
- **`tests/test_data/test_setup_path_equivalence.py`** — parametrized equivalence tests asserting that the general setup path (`dataset_type` in config) and the legacy `--modality` path produce byte-identical `processor_config.json` files, for all six modalities.
- **`tests/test_data/test_processor_signwriting.py`** — extended with `TestSignwritingModalityProcessorValidation`: verifies that `SignwritingModalityProcessor` raises `ValueError` immediately when `custom_preprocessor_path` is `None` or absent.
- **Golden file regression tests** — extended to cover `MultimodalMetaProcessor(slots=[...])` pipelines, verifying output is identical to the legacy wrappers.

#### CLI Verbosity Control

- **`--verbosity_level`** argument added to `multimodalhugs-train` and `multimodalhugs-generate`. Accepts `debug`, `info`, `warning` (default), `error`. Controls both the multimodalhugs package loggers and HuggingFace library loggers (transformers, datasets) with a single setting. At `warning` (default), the `[INFO|configuration_utils.py:...]` / `[INFO|modeling_utils.py:...]` clusters emitted during model and tokenizer loading are suppressed. At `info`, full HF verbosity is restored. `FutureWarning` messages from transformers are also suppressed at `warning` and above.
- The model architecture and parameter summary table are printed unconditionally via `print()` regardless of `verbosity_level`, preserving the pre-training architecture sanity check.

### Changed

- **`DataCollatorMultimodalSeq2Seq`** now detects whether its processor is a `MultimodalMetaProcessor` and, if so, delegates all processing (including label creation) to the processor's slots. The collator only adds `decoder_input_ids` from labels (via the model's `prepare_decoder_input_ids_from_labels`) when applicable. The legacy collation path (`_legacy_collate`) is retained for task-specific processors.
- **Label creation** is moved from the DataCollator into `TextModalityProcessor` with `role=TextRole.TARGET`. The collator no longer calls `create_seq2seq_labels_from_samples` when a `MultimodalMetaProcessor` is used.
- **`processors/__init__.py`** centralises all `AutoProcessor` registration. Previously, registration was scattered across individual setup scripts.
- **Legacy task-specific processors** (`Pose2TextTranslationProcessor`, etc.) are refactored into thin `MultimodalMetaProcessor` subclasses. Their constructors now build the same fixed four-slot list internally (modality slot + labels slot + encoder-prompt slot + decoder-input slot), delegating all logic to the parent class. Their `from_pretrained` methods detect a `"slots"` key in the saved config and delegate directly to `MultimodalMetaProcessor.from_pretrained`.
- **Legacy setup scripts** (`pose2text_training_setup.py`, etc.) updated to accept `dataset_type` in config and to call `build_processor_from_config` when a `pipeline:` or `slots:` key is present, falling back to the hardcoded path otherwise.
- **`DataCollatorMultimodalSeq2Seq`** now logs at `INFO` level (previously `WARNING`) when `return_tensors` is not `"pt"`, since this is an expected configuration in some inference scenarios.

### Deprecated

The following six task-specific processor classes are deprecated in favour of `MultimodalMetaProcessor` with explicit slot configuration. They remain fully functional and are not scheduled for removal in the near term, but new code should use the slot-based API.

| Deprecated class | Replacement |
|---|---|
| `Pose2TextTranslationProcessor` | `MultimodalMetaProcessor` with `PoseModalityProcessor` slot |
| `Video2TextTranslationProcessor` | `MultimodalMetaProcessor` with `VideoModalityProcessor` slot |
| `Features2TextTranslationProcessor` | `MultimodalMetaProcessor` with `FeaturesModalityProcessor` slot |
| `Image2TextTranslationProcessor` | `MultimodalMetaProcessor` with `ImageModalityProcessor` slot |
| `SignwritingProcessor` | `MultimodalMetaProcessor` with `SignwritingModalityProcessor` slot |
| `Text2TextTranslationProcessor` | `MultimodalMetaProcessor` with `TextModalityProcessor` slots |

### Fixed

- Fixed a typo in `FeaturesModalityProcessor`: the constructor parameter `temporal_dimention_position` is renamed to `temporal_dimension_position` (correct English spelling). The old name is not preserved. Saved processor configs using the old spelling must be updated by re-saving with `save_pretrained`.
- Fixed a typo in the `SignwritingProcessor` `name` attribute and `AutoProcessor` registry key: `"signwritting_processor"` → `"signwriting_processor"`. Saved processor configs referencing the old key must be re-saved with `save_pretrained` to enable `AutoProcessor.from_pretrained` resolution.
- Fixed hardcoded SignWriting string substitutions in `SignwritingModalityProcessor` that incorrectly normalised sign symbols, potentially altering the meaning of input sequences.
- Fixed `features2text` preset column map in `expand_pipeline_shorthand`: the map previously included `signal_start` and `signal_end` columns, which are not present in standard features metadata TSVs and caused `FeaturesModalityProcessor.process_sample` to receive an unexpected dict argument. The preset now maps only `{"signal": "signal"}`.
- Fixed a tokenizer cache double-extension bug in `build_processor_from_config`: a previous implementation cached the already-extended tokenizer and injected it into `TextModalityProcessor.__init__`, which then called `extend_tokenizer` a second time, producing an empty `new_tokens` list and corrupting the `pretrained_tokenizer` reference used for model construction. The cache now stores only the base (unextended) tokenizer.
- Fixed channel-count validation in `ImageModalityProcessor._load_from_path`: raises `ValueError` when the number of channels in the loaded image does not match the length of the `mean`/`std` normalization vectors.
- Fixed `_serialize_slot` in `MultimodalMetaProcessor` to include `custom_preprocessor` in its `_SKIP` set, preventing a spurious warning that the attribute "will be missing after `from_pretrained`". The attribute is correctly reconstructed from `custom_preprocessor_path` during loading.
- Fixed `print()` calls in `tokenizer_utils.add_new_special_tokens_from_vocab_file` replaced with `logger.info()`, so token-addition messages (added tokens, skipped tokens, save path) are routed through the logging system and suppressed at the default verbosity level.
- Fixed `MultimodalMetaProcessor.from_pretrained` to avoid redundant `AutoTokenizer.from_pretrained` + `extend_tokenizer` calls when multiple `TextModalityProcessor` slots share the same `(tokenizer_path, new_vocabulary)`. The tokenizer saved alongside the processor is pre-seeded into a cache keyed by `(tokenizer_path, new_vocabulary)`; subsequent slots with a matching key reuse the already-extended tokenizer and skip re-extension. See issue #78 for the related edge case where slots with different `tokenizer_path` values and no `new_vocabulary` silently receive the saved tokenizer.

## [0.4.1]

### Fixed

- Fixed a mutable default argument bug in `MultimodalSequence2SequenceProcessor.__call__` where `batch_dict={}` was shared across all calls within a process, causing silent data corruption when multiple processors ran in the same session.

### Added

- Added processor regression tests with golden values for all 6 modalities (pose, video, image, features, text, SignWriting), detecting unintended changes in preprocessing behaviour.
- Added dataset-to-processor contract tests verifying that each dataset's `_generate_examples()` output satisfies the keys and types expected by its processor.
- Added dataloader and datacollator tests covering batching, padding, and label construction.
- Added end-to-end overfitting test that trains a small model to convergence and asserts 100% ChrF on both validation and generation.

## [0.4.0]

### Fixed

- Fixed an issue where the maximum generation length was not properly configured, leading to truncated translations.
- Fixed tests that could not run in isolation before because of global variables.

### Added

- Added a parameter `use_backbone_max_length` for `MultimodalEmbedderConfig`.
- Added configuration tests.

### Changed

- Changed allowed `max_length` and `num_beams` parameters:
  - for `multimodalhugs-train`: `generation_max_length` and `generation_num_beams` are expected
  - for `multimodalhugs-generate`: `--max_length` and `--num_beams` are expected
