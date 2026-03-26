# Training Setup

`multimodalhugs-setup` (alias `mmhugs-setup`) initialises all artifacts required before training: the HuggingFace dataset cache, the processor, and the model weights.

---

## Usage

### General path (recommended)

No `--modality` argument required. The dataset class is selected from `data.dataset_type` in your YAML config:

```bash
multimodalhugs-setup --config_path config.yaml
```

### Legacy path (backward compatible)

Explicit `--modality` flag. Kept for configs that predate `data.dataset_type`. The flag is passed through to the corresponding task-specific setup script, which uses the general setup internally and falls back to the specified modality if `dataset_type` is not in the config:

```bash
multimodalhugs-setup --modality pose2text --config_path config.yaml
```

---

## CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config_path` | str | required | Path to the OmegaConf YAML config file |
| `--modality` | str | `null` | Legacy modality override (omit for general path) |
| `--do_dataset` | bool | `false` | Prepare and cache the HuggingFace dataset |
| `--do_processor` | bool | `false` | Build and save the processor |
| `--do_model` | bool | `false` | Build and save the model weights |
| `--output_dir` | str | `null` | Output directory (overrides `setup.output_dir` in config) |
| `--update_config` | bool | `null` | Write artifact paths back into the config file |
| `--rebuild_dataset_from_scratch` | bool | `false` | Ignore HuggingFace cache and rebuild dataset from zero |
| `--seed` | int | `42` | Random seed |

If none of `--do_dataset`, `--do_processor`, `--do_model` are set, all three steps are run.

---

## `data.dataset_type`

Add `dataset_type:` under the `data:` section of your config to tell the general setup which dataset class to use:

```yaml
data:
  dataset_type: pose2text     # see table below
  train_metadata_file: /path/to/train.tsv
  validation_metadata_file: /path/to/val.tsv
  test_metadata_file: /path/to/test.tsv
```

### Supported values

| `dataset_type` | Dataset class | Input modality |
|---|---|---|
| `pose2text` | `Pose2TextDataset` | Pose sequences (`.pose` files) |
| `video2text` | `Video2TextDataset` | Video files |
| `features2text` | `Features2TextDataset` | Pre-computed feature files (`.npy`) |
| `signwriting2text` | `SignWritingDataset` | ASCII SignWriting (FSW) strings |
| `image2text` | `BilingualImage2TextDataset` | Image files or text-rendered-as-image |
| `text2text` | `BilingualText2TextDataset` | Plain text (bilingual MT) |

The `dataset_type` values match the `--modality` values accepted by the legacy path.

---

## Config sections consumed by setup

### `setup:`

```yaml
setup:
  output_dir: /path/to/output   # required if not passed via --output_dir
  do_dataset: true
  do_processor: true
  do_model: true
  update_config: false          # write artifact paths back into this YAML
```

### `processor:`

The processor section is required when `do_processor: true`. See
[docs/processors/processor_config_formats.md](processors/processor_config_formats.md)
for the supported formats (`pipeline:` shorthand or `slots:` full format).

### `data:`

```yaml
data:
  dataset_type: pose2text           # selects dataset class (see table above)
  train_metadata_file: /path/to/train.tsv
  validation_metadata_file: /path/to/val.tsv
  test_metadata_file: /path/to/test.tsv
  shuffle: true
  max_frames: 300                   # modality-specific; not all datasets use this
```

### `model:`

```yaml
model:
  type: multimodal_embedder
  backbone_type: m2m_100
  pretrained_backbone: facebook/m2m100_418M
  multimodal_mapper_type: linear
  multimodal_mapper_dropout: 0.1
  feat_dim: 534
```

---

## Output artifacts

After a successful run, the output directory contains:

```
<output_dir>/
├── processor/          # saved MultimodalMetaProcessor
├── data/               # HuggingFace Dataset cache
├── model/              # model weights + config
└── actor_paths.yaml    # paths to all three artifacts (for training)
```

If `update_config: true`, the artifact paths are also written back into the config YAML for direct use with `multimodalhugs-train`.

---

## Full minimal example

```yaml
# config.yaml

data:
  dataset_type: pose2text
  train_metadata_file: /data/train.tsv
  validation_metadata_file: /data/val.tsv
  test_metadata_file: /data/test.tsv
  max_frames: 300

processor:
  pipeline: pose2text
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__asl__"

model:
  type: multimodal_embedder
  backbone_type: m2m_100
  pretrained_backbone: facebook/m2m100_418M
  multimodal_mapper_type: linear
  feat_dim: 534

setup:
  output_dir: /out/my_experiment
  update_config: true
```

```bash
multimodalhugs-setup --config_path config.yaml
```
