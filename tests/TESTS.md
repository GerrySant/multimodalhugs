# Test Suite Reference

This document catalogs every test in the suite, organized by file. Each entry describes what the test verifies.

---

## `test_config/`

### `test_multimodal_embedder_config.py`

| Test | What it checks |
|---|---|
| `test_config_max_length_default` | `MultiModalEmbedderConfig` has default `max_length=200` |
| `test_config_max_length_nondefault` | Custom `max_length` can be set at construction |
| `test_config_use_backbone_max_length_true` | `use_backbone_max_length=True` sets `max_length` to the backbone's value (512) |
| `test_config_use_backbone_max_length_fails_without_backbone_config` | `use_backbone_max_length=True` without a backbone config raises `ValueError` |

---

## `test_data/`

### `test_data_utils.py`

**`TestStringToList`** — `string_to_list()` parsing helper

| Test | What it checks |
|---|---|
| `test_valid_list_string` | `"[1, 2, 3]"` parses to `[1, 2, 3]` |
| `test_valid_nested_list` | `"[[1, 2], [3, 4]]"` parses to a nested list |
| `test_invalid_string_returns_none` | A non-list string returns `None` |
| `test_empty_list` | `"[]"` parses to `[]` |
| `test_float_list` | `"[0.5, 0.5, 0.5]"` parses to floats |

**`TestPadAndCreateMask`** — `pad_and_create_mask()` padding utility

| Test | What it checks |
|---|---|
| `test_same_length_tensors` | Equal-length tensors produce padded shape `(2, 5, 3)` and mask `(2, 5)` |
| `test_different_length_tensors` | Shorter tensors padded with zeros; mask tracks real positions |
| `test_single_tensor` | Single tensor produces correct shape `(1, 4, 2)` and mask `(1, 4)` |

**`TestCheckColumns`** — column presence validation

| Test | What it checks |
|---|---|
| `test_dataframe_all_present` | Returns `True` when all required columns present in DataFrame |
| `test_dataframe_missing_col` | Returns `False` when a column is missing |
| `test_hf_dataset_all_present` | Returns `True` for HF Dataset with all columns |
| `test_hf_dataset_missing_col` | Returns `False` for HF Dataset missing a column |

**`TestContainsEmpty`** — empty-value detection

| Test | What it checks |
|---|---|
| `test_empty_string` | Returns `True` for dict with empty string value |
| `test_none_value` | Returns `True` for dict with `None` value |
| `test_all_populated` | Returns `False` when all values are populated |

**`TestFileExistsFilter`** — file existence check

| Test | What it checks |
|---|---|
| `test_existing_file` | Returns `True` for an existing file |
| `test_nonexistent_file` | Returns `False` for a missing file |

**`TestDurationFilter`** — frame-count range filtering

| Test | What it checks |
|---|---|
| `test_no_bounds` | Returns `True` when no bounds specified |
| `test_within_bounds` | Returns `True` for duration within `min_frames`/`max_frames` |
| `test_below_min` | Returns `False` for duration below `min_frames` |
| `test_above_max` | Returns `False` for duration above `max_frames` |
| `test_only_min` | Returns `True` with only `min_frames` bound |
| `test_only_max` | Returns `True` with only `max_frames` bound |
| `test_at_exact_min` | Returns `True` at exactly `min_frames` |
| `test_at_exact_max` | Returns `True` at exactly `max_frames` |

**`TestSplitSentence`**, **`TestCreateImage`**, **`TestNormalizeImages`**, **`TestGetImages`** — image/text utility functions

| Test | What it checks |
|---|---|
| `test_basic_split` | `"Hello world"` splits into `["Hello", "world"]` |
| `test_punctuation_handling` | Punctuation preserved as tokens |
| `test_output_shape` | `create_image` produces shape `(224, 224, 3)` |
| `test_custom_size` | Custom size produces correct output shape |
| `test_normalized_values` | Normalized images fall in `[-1.1, 1.1]` |
| `test_multi_word_string` | One image per word, output shape `(N_words, C, H, W)` |

**`TestGatherAppropriateDataCfg`**, **`TestGetAllDataclassFields`**, **`TestBuildMergedOmegaconfConfig`**, **`TestResolveAndUpdateConfig`**, **`TestCenterImageOnWhiteBackground`** — config and layout utilities

| Test | What it checks |
|---|---|
| `test_obj_with_data_attr` | Extracts `data` attribute |
| `test_obj_with_dataset_attr` | Extracts `dataset` attribute |
| `test_plain_obj_fallback` | Returns plain object when no known attributes |
| `test_none_returns_empty_dict` | Returns `{}` for `None` input |
| `test_dict_with_data_key` | Extracts `data` key from dict |
| `test_multimodal_data_config_fields` | Finds inherited and own fields in `MultimodalDataConfig` |
| `test_inherited_fields_from_subclass` | Finds inherited fields in `Pose2TextDataConfig` |
| `test_non_dataclass_returns_empty` | Returns empty set for non-dataclass |
| `test_valid_keys_kept` | Preserves valid dataclass keys |
| `test_extra_keys_separated` | Separates extra keys into a separate dict |
| `test_overrides_take_precedence` | Kwargs override config values |
| `test_config_none_creates_new` | Creates new config when `None` passed |
| `test_existing_config_gets_updated` | Updates existing config in place |
| `test_non_config_kwargs_returned` | Returns non-matching kwargs separately |
| `test_output_size_matches_target` | Centering an image produces the target size `(100, 100)` |
| `test_different_target_sizes` | Different targets produce different output sizes |

---

### `test_dataset_configs.py`

**`TestMultimodalDataConfig`**, **`TestBilingualText2textMTDataConfig`**, **`TestPose2TextDataConfig`**, **`TestVideo2TextDataConfig`**, **`TestFeatures2TextDataConfig`**, **`TestBilingualImage2textMTDataConfig`**, **`TestConfigFromOmegaConf`** — dataset config dataclasses

| Test | What it checks |
|---|---|
| `test_defaults` (each class) | Default field values are correct |
| `test_custom_values_via_omegaconf` | OmegaConf overrides apply correctly |
| `test_from_omegaconf` | Config loads from an OmegaConf object |
| `test_custom_max_source_tokens` | `max_source_tokens` can be set |
| `test_custom_frame_limits` | `min_frames`/`max_frames` can be set |
| `test_preload_features` | `preload_features` flag is stored |
| `test_inherited_defaults` | Subclass inherits parent fields |
| `test_pose_config_from_omega` | Pose config loads from OmegaConf |
| `test_kwargs_override_cfg` | Kwargs override OmegaConf values |

---

### `test_dataset_bilingual_text2text.py`

**`TestBilingualText2TextDatasetInfo`** — dataset schema

| Test | What it checks |
|---|---|
| `test_info_returns_correct_features` | `_info()` includes `signal`, `encoder_prompt`, `decoder_prompt`, `output` |

**`TestBilingualText2TextSplitGenerators`** — split construction

| Test | What it checks |
|---|---|
| `test_all_metadata_files` | Returns 3 splits when all metadata files provided |
| `test_only_train` | Returns 1 split when only train metadata provided |
| `test_no_metadata_files` | Returns 0 splits when no metadata provided |

**`TestBilingualText2TextGenerateExamples`** — example generation

| Test | What it checks |
|---|---|
| `test_yields_correct_count` | Yields 3 examples from fixture |
| `test_yields_correct_keys` | Examples have all required keys |
| `test_correct_values` | First example has the expected field values |
| `test_missing_prompts_default_to_empty` | Missing prompts default to empty string |

**`TestBilingualText2TextMaxSourceTokensFilter`** — token-count filtering

| Test | What it checks |
|---|---|
| `test_max_source_tokens_filters_long_samples` | `max_source_tokens=1` removes multi-word samples |
| `test_max_source_tokens_keeps_short_samples` | `max_source_tokens=5` keeps all samples |

---

### `test_dataset_bilingual_image2text.py`

| Test | What it checks |
|---|---|
| `test_info_returns_correct_features` | `_info()` includes all required feature fields |
| `test_all_splits` | `_split_generators` returns 3 splits |
| `test_yields_correct_count` | Yields 2 examples from fixture |
| `test_yields_correct_fields` | Each example contains required fields |
| `test_signal_path_preserved` | Signal is an image file path ending in `.png` |

---

### `test_dataset_pose2text.py`

| Test | What it checks |
|---|---|
| `test_info_returns_correct_features` | `_info()` includes all required feature fields |
| `test_all_splits` | Returns 3 splits |
| `test_no_metadata` | Returns 0 splits when no metadata provided |
| `test_yields_correct_count` | Yields 3 examples |
| `test_yields_correct_fields` | Examples have all required fields |
| `test_file_exists_filter_removes_missing` | Missing pose files filtered out |
| `test_max_frames_filtering_train` | `max_frames=5` removes 10-frame samples from train split |
| `test_max_frames_no_filtering_if_large_enough` | `max_frames=100` keeps all samples |
| `test_test_split_skips_duration_filtering` | Test split ignores duration filter |

---

### `test_dataset_video2text.py`

| Test | What it checks |
|---|---|
| `test_info_returns_correct_features` | `_info()` includes all required feature fields |
| `test_all_splits` | Returns 3 splits |
| `test_no_metadata` | Returns 0 splits when no metadata provided |
| `test_yields_correct_count` | Yields 2 examples |
| `test_yields_correct_fields` | Examples have all required fields |
| `test_file_exists_filter` | Missing video files filtered out |
| `test_frame_count_works` | Frame counting via `av` works on dummy video |

---

### `test_dataset_features2text.py`

| Test | What it checks |
|---|---|
| `test_info_returns_correct_features` | `_info()` includes all required feature fields |
| `test_all_splits` | Returns 3 splits |
| `test_yields_correct_count` | Yields 3 examples |
| `test_yields_correct_fields` | Examples have all required fields |
| `test_file_exists_filter` | Missing `.npy` files filtered out |
| `test_max_frames_filtering` | `max_frames=5` removes 10-frame arrays |
| `test_max_frames_keeps_within_range` | `max_frames=100` keeps all samples |
| `test_preload_features_false_keeps_path` | `preload_features=False` keeps signal as file path |

---

### `test_dataset_signwriting.py`

| Test | What it checks |
|---|---|
| `test_info_returns_correct_features` | `_info()` includes all required feature fields |
| `test_all_splits` | Returns 3 splits |
| `test_yields_correct_count` | Yields 3 examples |
| `test_yields_correct_fields` | Examples have all required fields |
| `test_signal_is_fsw_string` | Signal is a valid FSW string starting with `M` |
| `test_no_file_filtering` | All rows yielded (no file I/O filtering) |

---

### `test_dataset_processor_contract.py`

Verifies that each dataset's output types and field names match what its corresponding processor expects.

| Class | What it checks |
|---|---|
| `TestPose2TextDatasetProcessorContract` | Signal is an existing file path; types are correct |
| `TestVideo2TextDatasetProcessorContract` | Signal is an existing file path; types are correct |
| `TestFeatures2TextDatasetProcessorContract` | Signal is an existing file path; types are correct |
| `TestText2TextDatasetProcessorContract` | Signal is a non-empty string; types are correct |
| `TestImage2TextDatasetProcessorContract` | Signal matches TSV text content; types are correct |
| `TestSignWritingDatasetProcessorContract` | Signal is a non-empty FSW string; types are correct |

Each class contains `test_dataset_yields_required_keys_and_types` and a modality-specific field check.

---

### `test_modality_processors.py`

Unit tests for all `ModalityProcessor` subclasses at the method level.

**`TestModalityProcessorBase`**

| Test | What it checks |
|---|---|
| `test_process_sample_default_is_noop` | Base class `process_sample` returns input unchanged |
| `test_process_batch_must_be_implemented` | `process_batch` not implemented raises `TypeError` |

**`TestPoseModalityProcessorProcessSample`**

| Test | What it checks |
|---|---|
| `test_reads_pose_file` | Loads `.pose` file to a 2D tensor `(frames, features)` |
| `test_tensor_passthrough` | Already-tensor input returned unchanged |
| `test_skip_frames_stride_reduces_temporal_dim` | `stride=2` halves frame count |
| `test_reduce_holistic_false_has_more_features` | Non-reduced pose has more feature dimensions |

**`TestPoseModalityProcessorProcessBatch`**

| Test | What it checks |
|---|---|
| `test_returns_tuple_of_two_tensors` | Returns `(data, mask)` |
| `test_data_is_3d` | Data is `(batch, frames, features)` |
| `test_mask_is_2d` | Mask is `(batch, frames)` |
| `test_variable_length_sequences_are_padded` | Variable-length sequences padded to max length |
| `test_mask_is_binary` | Mask contains only `0.0` and `1.0` |
| `test_data_and_mask_seq_dim_match` | Data and mask sequence dimensions are equal |

**`TestVideoModalityProcessorProcessSample`** / **`TestVideoModalityProcessorProcessBatch`**

| Test | What it checks |
|---|---|
| `test_reads_video_file` | Loads video to a tensor with `ndim >= 3` |
| `test_tensor_passthrough` | Tensor input returned unchanged |
| `test_returns_tuple_of_two_tensors` | Returns `(data, mask)` |
| `test_data_batch_dim_matches_input` | Batch dimension preserved |
| `test_variable_length_sequences_are_padded` | Variable frame counts padded |

**`TestTextModalityProcessorInputRole`**

| Test | What it checks |
|---|---|
| `test_process_batch_returns_tuple` | Returns `(ids, mask)` for `TextRole.INPUT` |
| `test_process_batch_batch_dim` | Batch dimension matches input size |
| `test_process_batch_ids_and_mask_same_shape` | `ids` and `mask` have equal shapes |
| `test_process_batch_pads_variable_length` | Variable-length text padded to same length |
| `test_process_batch_is_deterministic` | Two `TextRole.INPUT` processors produce identical output |

**`TestTextModalityProcessorTargetRole`**

| Test | What it checks |
|---|---|
| `test_returns_tensor` | Returns a tensor for `TextRole.TARGET` |
| `test_mask_is_none_for_labels` | Mask is `None` for target role |
| `test_batch_dim` | Batch dimension correct |
| `test_eos_token_is_last_real_token` | Last non-padding token is EOS |
| `test_target_prefix_appears_before_target` | `target_prefix` tokens precede `target` tokens |
| `test_shorter_sequence_padded_with_minus_100` | Short sequences padded with `-100` |
| `test_missing_output_returns_none` | Missing `target` field returns `(None, None)` |

---

### `test_processor_pose2text.py`

End-to-end tests for `Pose2TextTranslationProcessor`.

| Class | Tests |
|---|---|
| `TestPoseFileToTensor` | Loads `.pose` to tensor; tensor passthrough; `skip_frames_stride`; non-reduced feature count |
| `TestPoseObtainMultimodalInputAndMasks` | Returns 3D `input_frames` and 2D `attention_mask` |
| `TestPoseTransformGetItemsOutput` | `_transform_get_items_output` converts path to tensor |
| `TestPoseProcessorCall` | Returns `BatchFeature`; has all expected keys; batch dims consistent |

---

### `test_processor_video2text.py`

End-to-end tests for `Video2TextTranslationProcessor`.

| Class | Tests |
|---|---|
| `TestVideoFileToTensor` | Loads video to 4D tensor; tensor/ndarray passthrough; `skip_frames_stride` |
| `TestVideoObtainMultimodalInputAndMasks` | Returns `input_frames` and `attention_mask`; `join_chw` flattens to `[B, T, C*H*W]` |
| `TestVideoTransformGetItemsOutput` | Converts paths to tensors |
| `TestVideoProcessorCall` | Returns `BatchFeature`; has expected keys; batch dims consistent |

---

### `test_processor_features2text.py`

End-to-end tests for `Features2TextTranslationProcessor`.

| Class | Tests |
|---|---|
| `TestFeaturesFileToTensor` | Loads `.npy`; converts ndarray/list/tensor; `skip_frames_stride`; `temporal_dimension_position` |
| `TestFeaturesObtainMultimodalInputAndMasks` | Returns `input_frames` and `attention_mask`; variable-length padding |
| `TestFeaturesTransformGetItemsOutput` | Converts paths to tensors |
| `TestFeaturesCacheBehavior` | `use_cache=True` initializes internal cache |
| `TestFeaturesProcessorCall` | Returns `BatchFeature`; has expected keys; batch dims consistent |

---

### `test_processor_image2text.py`

End-to-end tests for `Image2TextTranslationProcessor`.

| Class | Tests |
|---|---|
| `TestImageToTensor` | Loads PNG/`.npy`; converts ndarray/tensor; renders text strings to images; normalization; raises on missing `mean`/`std` |
| `TestImageObtainMultimodalInputAndMasks` | Returns `input_frames` and `attention_mask` |
| `TestImageTransformGetItemsOutput` | Converts signals to tensors |
| `TestImageProcessorCall` | Returns `BatchFeature`; has expected keys; batch dims consistent |

---

### `test_processor_signwriting.py`

End-to-end tests for `SignwritingProcessor`.

| Class | Tests |
|---|---|
| `TestAsciiToTensor` | Converts FSW string to 4D tensor `[N_symbols, C, W, H]`; tensor passthrough |
| `TestSignwritingObtainMultimodalInputAndMasks` | Returns `input_frames` and `attention_mask`; variable-length padding |
| `TestSignwritingTransformGetItemsOutput` | Converts FSW strings to tensors |
| `TestSignwritingProcessorCall` | Returns `BatchFeature`; has expected keys; batch dims consistent |

**`TestSignwritingModalityProcessorValidation`** — required-argument validation on `SignwritingModalityProcessor`

| Test | What it checks |
|---|---|
| `test_raises_when_no_preprocessor_path` | `SignwritingModalityProcessor()` with no arguments raises `ValueError` mentioning `custom_preprocessor_path` |
| `test_raises_with_none_preprocessor_path` | `SignwritingModalityProcessor(custom_preprocessor_path=None)` raises `ValueError` mentioning `custom_preprocessor_path` |

---

### `test_processor_text2text.py`

End-to-end tests for `Text2TextTranslationProcessor`.

| Class | Tests |
|---|---|
| `TestText2TextObtainMultimodalInputAndMasks` | Returns `input_ids` and `attention_mask`; shapes match; variable-length padding |
| `TestText2TextObtainPrompts` | Returns `encoder_prompt` with mask; `decoder_input_ids` with mask |
| `TestText2TextProcessorCall` | Returns `BatchFeature`; has all expected keys |
| `TestText2TextProcessPrompts` | Encoder and decoder slots tokenize text |

---

### `test_meta_processor.py`

Tests for `ProcessorSlot` and `MultimodalMetaProcessor` (the flat-slots architecture).

**`TestProcessorSlot`** — slot dataclass behaviour

| Test | What it checks |
|---|---|
| `test_instantiation_with_required_fields` | Slot stores `processor` and `output_data_key`; default `primary_field` is `"signal"` |
| `test_output_mask_key_defaults_to_none` | `output_mask_key` defaults to `None` |
| `test_output_mask_key_can_be_set` | `output_mask_key` can be set |
| `test_primary_field_custom_column_map` | `primary_field` is derived from the first key of `column_map` |
| `test_is_label_default_false` | `is_label` defaults to `False` |
| `test_is_label_true` | `is_label=True` is stored correctly |

**`TestMultimodalMetaProcessorPose2Text`** — pose→text pipeline via flat slots

| Test | What it checks |
|---|---|
| `test_call_returns_batch_feature` | `__call__` returns `BatchFeature` |
| `test_call_produces_input_frames` | Output contains `input_frames` |
| `test_call_produces_attention_mask` | Output contains `attention_mask` |
| `test_call_produces_encoder_prompt` | Output contains `encoder_prompt` and `encoder_prompt_length_padding_mask` |
| `test_call_produces_decoder_input_ids` | Output contains `decoder_input_ids` and `decoder_attention_mask` |
| `test_call_produces_labels` | Output contains `labels` |
| `test_input_frames_shape` | `input_frames` is `(batch, frames, features)` |
| `test_attention_mask_shape` | `attention_mask` is `(batch, frames)` |
| `test_labels_shape` | `labels` is `(batch, seq_len)` |
| `test_all_batch_dims_consistent` | All tensors share the same batch dimension |
| `test_transform_get_items_output_converts_path_to_tensor` | `_transform_get_items_output` converts file paths to tensors |
| `test_transform_get_items_output_leaves_other_columns_intact` | Text columns not modified by transform |
| `test_label_slot_is_marked` | Exactly one slot has `is_label=True` |

**`TestMultimodalMetaProcessorText2Text`** — text→text pipeline via flat slots

| Test | What it checks |
|---|---|
| `test_call_returns_batch_feature` | Returns `BatchFeature` |
| `test_call_produces_input_ids` | Output contains `input_ids` |
| `test_call_produces_attention_mask` | Output contains `attention_mask` |
| `test_call_produces_labels` | Output contains `labels` |
| `test_all_batch_dims_consistent` | All tensors share the same batch dimension |

**`TestMultimodalMetaProcessorMultiInput`** — two-encoder-stream pipeline (video + pose)

| Test | What it checks |
|---|---|
| `test_call_produces_video_frames` | Output contains `video_frames` |
| `test_call_produces_video_mask` | Output contains `video_attention_mask` |
| `test_call_produces_pose_frames` | Output contains `pose_frames` |
| `test_call_produces_pose_mask` | Output contains `pose_attention_mask` |
| `test_call_produces_labels` | Output contains `labels` |
| `test_output_keys_match_slot_declarations` | Every declared `output_data_key`/`output_mask_key` present in output |
| `test_video_and_pose_masks_are_independent` | Video and pose masks are separate tensors |
| `test_all_batch_dims_consistent` | All tensors share the same batch dimension |

**`TestMetaProcessorBackwardCompatibility`** — output key parity with legacy processors

| Test | What it checks |
|---|---|
| `test_pose2text_produces_all_legacy_keys` | Produces all keys that `Pose2TextTranslationProcessor` + DataCollator produced |
| `test_text2text_produces_all_legacy_keys` | Produces all keys that `Text2TextTranslationProcessor` + DataCollator produced |

**`TestDataCollatorWithMetaProcessor`** — collator delegates label creation to the processor

| Test | What it checks |
|---|---|
| `test_collator_can_be_instantiated_without_tokenizer` | DataCollator works without a tokenizer argument |
| `test_collator_output_contains_labels` | Output contains `labels` |
| `test_collator_output_contains_input_frames` | Output contains `input_frames` |
| `test_collator_batch_size_preserved` | Batch dimensions correct |
| `test_collator_labels_come_from_meta_not_collator` | Labels present even when collator has no tokenizer |

**`TestMetaProcessorConstruction`** — flat-slots construction

| Test | What it checks |
|---|---|
| `test_slots_stored_in_order` | Slots stored in declaration order |
| `test_label_slot_is_labeled` | Exactly one slot has `is_label=True` |
| `test_tokenizer_stored` | Tokenizer stored on the processor |

**`TestMetaProcessorSerialization`** — `to_dict()` output

| Test | What it checks |
|---|---|
| `test_to_dict_has_slots` | `to_dict()` contains a `"slots"` key |
| `test_to_dict_slot_has_required_keys` | Each slot dict has `processor_class`, `processor_kwargs`, `output_data_key`, `is_label`, `column_map` |
| `test_to_dict_label_slot_is_label_true` | Label slot has `"is_label": true` in dict |
| `test_to_dict_is_json_serializable` | `to_dict()` output can be serialized to JSON |

**`TestMetaProcessorSavePretrained`** — save/load round-trip (basic)

| Test | What it checks |
|---|---|
| `test_roundtrip_slot_count` | Slot count preserved after save/load |
| `test_roundtrip_output_keys` | `output_data_key`, `output_mask_key`, `is_label` preserved |
| `test_roundtrip_column_maps` | `column_map` preserved |
| `test_roundtrip_processor_classes` | Processor class names preserved |
| `test_roundtrip_output_consistent` | Same input produces identical output before and after save/load |

**`TestMetaProcessorCall`** — `__call__` behaviour

| Test | What it checks |
|---|---|
| `test_returns_batch_feature` | Returns `BatchFeature` |
| `test_has_expected_keys` | All declared output keys present |
| `test_batch_dimensions_consistent` | All tensors have matching batch dimension |
| `test_prepopulated_key_is_not_overwritten` | Keys already in `batch_dict` are not overwritten |
| `test_labels_in_output` | Labels always present in output |

**`TestMetaProcessorTransform`** — `_transform_get_items_output` behaviour

| Test | What it checks |
|---|---|
| `test_tensor_written_back` | File paths converted to tensors in-place |
| `test_text_not_corrupted` | Text columns not altered by the transform |
| `test_missing_primary_field_skipped` | Slots whose `primary_field` is absent are skipped and a `logger.warning` is emitted |

**`TestMetaProcessorValidation`** — constructor guard rails

| Test | What it checks |
|---|---|
| `test_empty_slots_raises` | `MultimodalMetaProcessor(slots=[])` raises `ValueError` |
| `test_duplicate_output_data_key_raises` | Two slots sharing `output_data_key` raise `ValueError` |
| `test_duplicate_output_mask_key_raises` | Two slots sharing a non-`None` `output_mask_key` raise `ValueError` |
| `test_none_mask_keys_do_not_conflict` | Multiple slots with `output_mask_key=None` do not raise |

**`TestFromPretrainedRegistry`** — `processor_registry` kwarg on `from_pretrained`

| Test | What it checks |
|---|---|
| `test_unknown_class_raises_attribute_error` | Unknown `processor_class` in saved JSON raises `AttributeError` with a hint about `processor_registry` |
| `test_registry_resolves_custom_class` | User-defined `ModalityProcessor` subclass passed via `processor_registry=` is resolved correctly |

**`TestMissingColumnWarning`** — warning for missing primary field

| Test | What it checks |
|---|---|
| `test_warning_emitted_for_missing_primary_field` | `_transform_get_items_output` emits a `logger.warning` mentioning the missing column name and the slot's `output_data_key` when the slot's `primary_field` is absent from the batch |

**`TestTokenizerCacheScenarios`** — tokenizer cache behaviour in `build_processor_from_config`

| Test | What it checks |
|---|---|
| `test_different_paths_produce_independent_tokenizers` | Two slots with different `tokenizer_path` get fully independent tokenizer objects |
| `test_same_path_same_vocab_produce_shared_base_tokenizer` | Two slots with the same `tokenizer_path` (and no `new_vocabulary`) share the same base tokenizer object via the cache |
| `test_same_path_different_vocab_emits_warning_and_produces_different_tokenizers` | Two slots with the same `tokenizer_path` but different `new_vocabulary` files produce independently-extended tokenizers with different vocabulary sizes, and a `logger.warning` is emitted |

**`TestMetaProcessorMultiSlot`** — multi-encoder-slot with features

| Test | What it checks |
|---|---|
| `test_multi_input_produces_all_keys` | Both encoder streams and prompts appear in the output |

**`TestMultimodalMetaProcessorRoundTrip`** — deep save/load equivalence

| Test | What it checks |
|---|---|
| `test_loaded_is_multimodal_meta_processor` | `from_pretrained` returns a `MultimodalMetaProcessor` instance |
| `test_text2text_slot_structure_preserved` | Slot types, keys, column maps, and `is_label` identical after load |
| `test_text2text_encoder_slot_processor_type` | Encoder slot holds a `TextModalityProcessor` after load |
| `test_text2text_output_identical` | All output tensors identical before and after save/load (text→text) |
| `test_text2text_transform_output_identical` | `_transform_get_items_output` output identical after load |
| `test_features2text_processor_kwargs_preserved` | `skip_frames_stride`, `temporal_dimension_position`, `use_cache` survive serialisation |
| `test_features2text_output_identical` | All output tensors identical before and after save/load (features→text) |

---

### `test_datacollator.py`

**`TestCreateSeq2SeqLabels`** — `create_seq2seq_labels_from_samples()` function

| Test | What it checks |
|---|---|
| `test_basic_label_creation` | Creates a `(2,)` batch labels tensor |
| `test_missing_output_returns_none` | Returns `None` when any `output` field is `None` |
| `test_padding_applied` | Sequences padded to same length; shorter padded with `-100` |
| `test_no_padding_returns_raw_lists` | `padding=False` returns lists, not tensors |
| `test_max_length_padding` | `MAX_LENGTH` padding with `max_length=20` produces `shape[1]==20` |
| `test_pad_to_multiple_of` | `pad_to_multiple_of=8` produces `shape[1]` divisible by 8 |
| `test_eos_token_appended` | Last token is the tokenizer's EOS token |
| `test_return_tensors_np` | `return_tensors="np"` produces a NumPy array |

**`TestDataCollatorInit`** — collator construction

| Test | What it checks |
|---|---|
| `test_init_with_processor_and_tokenizer` | Stores both processor and tokenizer |
| `test_init_tokenizer_none_falls_back` | Falls back to `processor.tokenizer` when `tokenizer=None` |

**`TestDataCollatorCall`** — collator with legacy processors

| Test | What it checks |
|---|---|
| `test_text2text_full_call` | Produces `input_ids`, `attention_mask`, `labels` |
| `test_features2text_full_call` | Produces `input_frames`, `attention_mask`, `labels` |
| `test_labels_batch_dimension_matches` | Labels and inputs share the same batch dimension |

**`TestDataCollatorWithMetaProcessor`** — collator delegates to `MultimodalMetaProcessor`

| Test | What it checks |
|---|---|
| `test_text2text_returns_expected_keys` | All expected output keys present |
| `test_features2text_returns_expected_keys` | All expected output keys present |
| `test_labels_are_tensors` | Labels are `torch.Tensor` |
| `test_labels_batch_dim_matches` | Labels batch dimension matches input |
| `test_labels_padded_with_minus_100` | All label values are either valid token IDs or `-100` |
| `test_meta_labels_match_legacy_labels` | Labels from MetaProcessor identical to legacy `create_seq2seq_labels_from_samples` output |
| `test_legacy_processor_path_still_works` | Old-style processors produce labels through the legacy collator path |

---

### `test_general_training_setup.py`

Tests for `_build_dataset_map()` and `general_training_setup.main()` in `training_setup/general_training_setup.py`.

**`TestBuildDatasetMapKeys`** — registry key names

| Test | What it checks |
|---|---|
| `test_returns_all_expected_keys` | Map contains exactly the 6 expected `dataset_type` keys |
| `test_no_legacy_signwriting_key` | `"signwriting"` (old name) is not present |
| `test_no_bilingual_image2text_key` | `"bilingual_image2text"` (old name) is not present |
| `test_no_bilingual_text2text_key` | `"bilingual_text2text"` (old name) is not present |

**`TestBuildDatasetMapValues`** — class references per dataset type

| Test | What it checks |
|---|---|
| `test_each_entry_is_two_tuple` | Every map value is a 2-tuple `(DatasetClass, DataConfigClass)` |
| `test_pose2text_classes` | `"pose2text"` maps to `Pose2TextDataset` / `Pose2TextDataConfig` |
| `test_video2text_classes` | `"video2text"` maps to `Video2TextDataset` / `Video2TextDataConfig` |
| `test_features2text_classes` | `"features2text"` maps to `Features2TextDataset` / `Features2TextDataConfig` |
| `test_signwriting2text_classes` | `"signwriting2text"` maps to `SignWritingDataset` / `MultimodalDataConfig` |
| `test_image2text_classes` | `"image2text"` maps to `BilingualImage2TextDataset` / `BilingualImage2textMTDataConfig` |
| `test_text2text_classes` | `"text2text"` maps to `BilingualText2TextDataset` / `BilingualText2textMTDataConfig` |

**`TestBuildDatasetMapMatchesModalityMap`** — consistency with CLI `--modality` values

| Test | What it checks |
|---|---|
| `test_all_modality_map_keys_covered` | `dataset_type` keys exactly mirror the CLI `--modality` accepted values |

**`TestGeneralTrainingSetupMainValidation`** — config validation errors

| Test | What it checks |
|---|---|
| `test_missing_dataset_type_raises` | `do_dataset=True` with no `data.dataset_type` in config raises `ValueError` mentioning `data.dataset_type` |
| `test_unknown_dataset_type_raises` | An unrecognised `dataset_type` value raises `ValueError` mentioning the bad value |
| `test_missing_processor_config_raises` | `do_processor=True` with no `processor:` section raises `ValueError` mentioning `processor` |
| `test_default_dataset_type_fallback` | `default_dataset_type` argument bypasses the "required" error when `data.dataset_type` is absent |

---

### `test_setup_path_equivalence.py`

Parametrized equivalence tests verifying that the legacy `--modality` path and the general path (with `dataset_type` in config) produce identical processor artifacts for all six supported modalities. Uses committed binary assets from `tests/assets/`. Each test is parametrized over `[pose2text, video2text, features2text, signwriting2text, image2text, text2text]`.

**`TestSetupPathEquivalence`** — 2 tests × 6 modalities = 12 tests total

| Test | What it checks |
|---|---|
| `test_processor_slot_structure_identical[<modality>]` | General path and legacy `--modality` path produce processors with identical slot structure (class names, output keys, column maps, is_label flags) |
| `test_processor_json_byte_identical[<modality>]` | `processor_config.json` is byte-for-byte identical between general and legacy paths |

---

### `test_setup_utils.py`

Tests for `build_processor_from_config()` and `expand_pipeline_shorthand()` in `training_setup/setup_utils.py`.

**`TestBuildProcessorFromConfigReturnsNone`** — no-op cases

| Test | What it checks |
|---|---|
| `test_returns_none_when_slots_absent` | Returns `None` when no `slots` key in config |
| `test_returns_none_when_slots_empty_list` | Returns `None` for an empty `slots` list |
| `test_returns_none_for_none_cfg` | Returns `None` when `cfg` is `None` |

**`TestBuildProcessorFromConfigReturnsProcessor`** — successful construction

| Test | What it checks |
|---|---|
| `test_returns_meta_processor` | Returns a `MultimodalMetaProcessor` instance |
| `test_slot_count_matches` | Number of slots matches the YAML declaration |
| `test_output_data_keys` | `input_ids` and `labels` keys present on slots |
| `test_output_mask_key` | `output_mask_key` set correctly on the encoder slot |
| `test_is_label_flag` | `is_label=True` on the labels slot |
| `test_non_label_slot_is_false` | `is_label=False` on non-label slots |
| `test_column_map_set` | `column_map` matches the YAML declaration |
| `test_processor_class_instantiated` | Each slot holds an instance of the declared processor class |
| `test_tokenizer_loaded_from_path` | `TextModalityProcessor.tokenizer` loaded from `tokenizer_path` |
| `test_meta_processor_tokenizer_auto_derived` | `MultimodalMetaProcessor.tokenizer` auto-derived from first text slot |

**`TestBuildProcessorFromConfigFeaturesSlot`** — non-text processor (no tokenizer param)

| Test | What it checks |
|---|---|
| `test_features_slot_built` | `FeaturesModalityProcessor` slot is constructed without error |
| `test_features_meta_tokenizer_is_none` | `MultimodalMetaProcessor.tokenizer` is `None` when there are no text slots |
| `test_features_processor_kwargs_forwarded` | Constructor kwargs (e.g. `skip_frames_stride`) forwarded to the processor |

**`TestBuildProcessorFromConfigDefaultColumnMap`**

| Test | What it checks |
|---|---|
| `test_default_column_map` | Slot without explicit `column_map` gets default `{"signal": "signal"}` |

**`TestBuildProcessorFromConfigProducesValidOutput`** — end-to-end batch processing

| Test | What it checks |
|---|---|
| `test_text_batch_produces_expected_keys` | Built processor produces `input_ids`, `attention_mask`, `labels` from a text batch |
| `test_text_batch_label_batch_dim` | Labels batch dimension matches number of input samples |

**`TestExpandPipelineShorthandPassthrough`** — non-shorthand configs returned unchanged

| Test | What it checks |
|---|---|
| `test_passthrough_none` | `None` input returned unchanged |
| `test_passthrough_slots_present` | Config with `slots:` key returned as the same object (no expansion) |
| `test_passthrough_legacy_config` | Config with neither `pipeline:` nor `slots:` returned unchanged |
| `test_passthrough_preserves_omegaconf_type` | OmegaConf type preserved on passthrough |

**`TestExpandPipelineShorthandValidation`** — error cases

| Test | What it checks |
|---|---|
| `test_unknown_pipeline_raises` | Unknown `pipeline:` value raises `ValueError` with informative message |
| `test_missing_tokenizer_path_raises` | Absent `tokenizer_path` raises `ValueError` |

**`TestExpandPipelineShorthandStructure`** — output shape and key hygiene

| Test | What it checks |
|---|---|
| `test_returns_slots_key` | Expanded config contains a `slots` key |
| `test_four_slots_generated` | Exactly 4 slots generated (1 modality + 3 text) |
| `test_output_data_keys_match_standard` | Output keys are `input_frames`, `labels`, `encoder_prompt`, `decoder_input_ids` |
| `test_pipeline_key_removed` | `pipeline:` key absent from expanded config |
| `test_shorthand_keys_removed` | All shorthand keys (`pipeline`, `tokenizer_path`, `new_vocabulary`, `modality_kwargs`) absent |
| `test_returns_omegaconf_for_omegaconf_input` | OmegaConf input → OmegaConf output |
| `test_returns_dict_for_dict_input` | Plain dict input → plain dict output |

**`TestExpandPipelineShorthandModalitySlot`** — first-slot processor class per pipeline

| Test | What it checks |
|---|---|
| `test_modality_processor_class[pose2text-PoseModalityProcessor]` | `pose2text` uses `PoseModalityProcessor` |
| `test_modality_processor_class[video2text-VideoModalityProcessor]` | `video2text` uses `VideoModalityProcessor` |
| `test_modality_processor_class[features2text-FeaturesModalityProcessor]` | `features2text` uses `FeaturesModalityProcessor` |
| `test_modality_processor_class[image2text-ImageModalityProcessor]` | `image2text` uses `ImageModalityProcessor` |
| `test_modality_processor_class[signwriting2text-SignwritingModalityProcessor]` | `signwriting2text` uses `SignwritingModalityProcessor` |
| `test_modality_processor_class[text2text-TextModalityProcessor]` | `text2text` uses `TextModalityProcessor` |
| `test_modality_slot_output_data_key` | Modality slot writes to `input_frames` |
| `test_modality_slot_output_mask_key` | Modality slot mask key is `attention_mask` |
| `test_pose_column_map_has_offsets` | `pose2text` column_map includes `signal_start` and `signal_end` |
| `test_image_column_map_no_offsets` | `image2text` column_map does not include temporal offsets |
| `test_modality_kwargs_forwarded` | `modality_kwargs` forwarded to modality slot's `processor_kwargs` |
| `test_no_processor_kwargs_when_modality_kwargs_absent` | No `processor_kwargs` set when `modality_kwargs` is absent |
| `test_text2text_modality_slot_has_tokenizer` | `text2text` modality slot injects `tokenizer_path` |
| `test_text2text_modality_slot_has_role_input` | `text2text` modality slot has `role=input` |

**`TestExpandPipelineShorthandTextSlots`** — standard text output slots

| Test | What it checks |
|---|---|
| `test_labels_slot_is_label_true` | `labels` slot has `is_label=True` |
| `test_labels_slot_column_map` | `labels` column_map is `{decoder_prompt: target_prefix, output: target}` |
| `test_labels_slot_role_target` | `labels` slot has `role=target` |
| `test_encoder_prompt_slot_has_mask_key` | `encoder_prompt` slot has correct mask key |
| `test_decoder_input_ids_slot_has_mask_key` | `decoder_input_ids` slot has correct mask key |
| `test_tokenizer_path_in_text_slots` | All three text slots contain `tokenizer_path` |
| `test_new_vocabulary_propagated` | `new_vocabulary` propagated to all text slots when set |
| `test_new_vocabulary_absent_when_not_set` | `new_vocabulary` absent from slot kwargs when not provided |

**`TestExpandPipelineShorthandEndToEnd`** — shorthand roundtrip through `build_processor_from_config`

| Test | What it checks |
|---|---|
| `test_pipeline_shorthand_builds_processor` | `build_processor_from_config` with `pipeline:` returns a `MultimodalMetaProcessor` |
| `test_pipeline_shorthand_slot_count` | Processor has 4 slots |
| `test_pipeline_shorthand_tokenizer_set` | `MultimodalMetaProcessor.tokenizer` is non-`None` |

---

### `test_optional_import_guards.py`

Tests that each modality processor and dataset raises a clear `ImportError` (mentioning the missing package) when its optional dependency is absent. Module-level `_*_AVAILABLE` flags are patched via `monkeypatch` to simulate a missing package without uninstalling anything.

**Processors**

| Test | What it checks |
|---|---|
| `test_pose_processor_raises_without_pose_format` | `PoseModalityProcessor()` raises `ImportError` mentioning `pose-format` when `_POSE_FORMAT_AVAILABLE=False` |
| `test_signwriting_processor_raises_without_signwriting` | `SignwritingModalityProcessor()` raises `ImportError` mentioning `signwriting` when `_SIGNWRITING_AVAILABLE=False` |
| `test_video_processor_raises_without_cv2_when_custom_preprocessor` | `VideoModalityProcessor(custom_preprocessor_path=...)` raises `ImportError` mentioning `opencv-python` when `_CV2_AVAILABLE=False` |
| `test_video_processor_raises_without_torchvision` | `VideoModalityProcessor()` raises `ImportError` mentioning `torchvision` when `_TORCHVISION_AVAILABLE=False` |
| `test_image_processor_raises_without_cv2` | `ImageModalityProcessor()` raises `ImportError` mentioning `opencv-python` when `_CV2_AVAILABLE=False` |

**Datasets**

| Test | What it checks |
|---|---|
| `test_pose2text_dataset_raises_without_pose_format` | `Pose2TextDataset()` raises `ImportError` mentioning `pose-format` when `_POSE_FORMAT_AVAILABLE=False` |
| `test_signwriting_dataset_raises_without_signwriting` | `SignWritingDataset()` raises `ImportError` mentioning `signwriting` when `_SIGNWRITING_AVAILABLE=False` |
| `test_video2text_dataset_raises_without_av` | `Video2TextDataset()` raises `ImportError` mentioning `av` when `_AV_AVAILABLE=False` |
| `test_video2text_dataset_raises_without_torchvision` | `Video2TextDataset()` raises `ImportError` mentioning `torchvision` when `_TORCHVISION_AVAILABLE=False` |
| `test_video_processor_no_raise_without_cv2_when_no_custom_preprocessor` | `VideoModalityProcessor(custom_preprocessor_path=None)` does **not** raise when `_CV2_AVAILABLE=False`; cv2 is only required when a custom preprocessor path is provided |

---

### `test_processor_regression.py`

Regression tests comparing processor output against golden files in `tests/assets/golden/`. Golden files capture shape, dtype, mean, std, min, max, sum, and first/last 8 values of every output tensor. To regenerate: `python tests/assets/generate_golden.py`.

**Legacy processor classes** — verify that existing processors have not changed behaviour:

| Class | Processor under test | Golden file |
|---|---|---|
| `TestPose2TextRegression` | `Pose2TextTranslationProcessor` | `pose2text.json` |
| `TestVideo2TextRegression` | `Video2TextTranslationProcessor` | `video2text.json` |
| `TestFeatures2TextRegression` | `Features2TextTranslationProcessor` | `features2text.json` |
| `TestText2TextRegression` | `Text2TextTranslationProcessor` | `text2text.json` |
| `TestLabelsRegression` | `create_seq2seq_labels_from_samples` | `labels.json` |
| `TestImage2TextRegression` | `Image2TextTranslationProcessor` | `image2text.json` |

**New flat-slots classes** — verify that `MultimodalMetaProcessor(slots=[...])` produces identical output to the legacy wrappers (same golden files):

| Class | Slot configuration | Golden file |
|---|---|---|
| `TestMetaProcessorPose2TextGolden` | `PoseModalityProcessor(reduce_holistic_poses=True)` | `pose2text.json` |
| `TestMetaProcessorVideo2TextGolden` | `VideoModalityProcessor(custom_preprocessor_path=..., use_cache=True)` | `video2text.json` |
| `TestMetaProcessorFeatures2TextGolden` | `FeaturesModalityProcessor(use_cache=False)` | `features2text.json` |
| `TestMetaProcessorText2TextGolden` | `TextModalityProcessor(role=TextRole.INPUT)` | `text2text.json` |
| `TestMetaProcessorImage2TextGolden` | `ImageModalityProcessor(font_path=..., width=224, height=224)` | `image2text.json` |

---

## `test_model_only/`

### `test_model_only.py`

Parametrized over three model configurations (default `max_length`, backbone-derived `max_length`, explicit `max_length`).

| Test | What it checks |
|---|---|
| `test_model_maxlength_is_correct` | Model `max_length` matches expected value for each config (200, 15, or 20) |
| `test_training` | Model overfits to a tiny batch: loss drops below `0.11` within 500 epochs |
| `test_overfitting_accuracy` | WER ≤ 0.125 on test samples after training on the same data |

---

## `e2e_overfitting/`

### `test_e2e_overfitting.py`

Full end-to-end pipeline test using the image2text modality.

| Test | What it checks |
|---|---|
| `test_setup_runs_successfully` | `multimodalhugs_cli.training_setup` completes without error |
| `test_model_converges_in_training` | Training run achieves `eval_chrf=100.0` |
| `test_generation_score_is_perfect` | Generation on the best checkpoint with `--metric_name sacrebleu,chrf` achieves `predict_sacrebleu=100.0` and `predict_chrf=100.0`; verifies both metric keys are present in `predict_results.json` (exercises the multi-metric zip loop and separate `evaluate.load` calls) |
