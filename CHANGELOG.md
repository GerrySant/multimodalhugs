# Changelog

All notable changes to the project are documented in this file.

Version numbers are of the form `1.0.0`.

Each version section may have subsections for: _Added_, _Changed_, _Removed_, _Deprecated_, and _Fixed_.

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
