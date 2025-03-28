<div align="center">
  <h1>🎨 MultiModalHugs</h1>
</div>

**MultimodalHugs** is a streamlined extension of [Hugging Face](https://huggingface.co/) designed for training, evaluating, and deploying multimodal AI models. Built atop Hugging Face’s powerful ecosystem, MultimodalHugs integrates seamlessly with standard pipelines while providing additional functionalities to handle multilingual and multimodal inputs—reducing boilerplate and simplifying your codebase.

---

## Key Features

- **Unified Framework**: Train and evaluate multimodal models (e.g., image-to-text, pose-to-text, signwriting-to-text) using a consistent API.
- **Minimal Code Changes**: Leverage Hugging Face’s pipelines with only minor modifications.
- **Data in TSV**: Avoid the complexity of numerous hyperparameters by maintaining data splits in `.tsv` files—easily specify prompts, languages, targets, or other attributes in dedicated columns.
- **Modular Design**: Use or extend any of the components (datasets, models, modules, processors) to suit your custom tasks.
- **Examples Included**: Refer to the `examples/` directory for guided scripts, configurations, and best practices.

For more details, refer to the [documentation](docs/README.md).

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/GerrySant/multimodalhugs.git
   ```

2. **Navigate and install the package**:

   - **Standard installation**:
      ```bash
       cd multimodalhugs
       pip install .
      ```
   - **Developer installation**:
      ```bash
       cd multimodalhugs
       pip install -e .[dev]
      ```

## Usage

### 🚀 Getting Started

To set up, train, and evaluate a model, follow these steps:

![Steps Overview](docs/media/steps.png)

### 1. Dataset Preparation

For each partition (train, val, test), create a TSV file that captures essential sample details (input paths, timestamps, prompts, target texts) for consistency.

#### Metadata File Requirements

The `metadata.tsv` files for each partition must include the following fields:

- `signal`: The signal text for the translation from which the images will be created / The path of the images to be uploaded (supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.npy`)
- `encoder_prompt`: A text string (e.g., `__vhe__`) that helps the model distinguish the signal language or modality. Can be empty if not used.
- `decoder_prompt`: A text prompt appended during decoding to guide the model’s generation. Useful for specifying style or language; can be empty if not used.
- `output`: The target text for translation.

### 2. Setup Datasets, Model, and Processors

```bash
multimodalhugs-setup --modality {pose2text,signwriting2text,image2text} --config_path CONFIG_PATH
```

### 3. Train a Model

```bash
multimodalhugs-train --task translation --config_path CONFIG_PATH
```

### 4. Generate Outputs with a Trained Model

```bash
multimodalhugs-generate --task translation --config_path CONFIG_PATH --model_name_or_path MODEL_PATH --processor_name_or_path PROCESSOR_PATH --dataset_dir DATASET_PATH --output_dir OUTPUT_DIR
```

For more details, refer to the [CLI documentation](docs/general/CLI.md).

[Here](/examples/multimodal_translation/) you can find some sample end-to-end experimentation pipelines.

## Directory Overview

```yaml
multimodalhugs/
├── README.md               # Project overview
├── LICENSE                 # License information
├── pyproject.toml          # Package dependencies and setup
├── docs/                   # Documentation
│   ├── data/               # Data-related documentation
│   ├── general/            # General framework documentation
│   ├── models/             # Model-related documentation
│   ├── media/              # Visual guides
│   └── README.md           # Documentation overview
├── examples/               # Example scripts and configurations
│   ├── multimodal_translation/
│   │   ├── image2text_translation/
│   │   ├── pose2text_translation/
│   │   └── signwriting2text_translation/
├── multimodalhugs/         # Core framework
│   ├── data/               # Data handling utilities
│   ├── models/             # Model implementations
│   ├── modules/            # Custom components (adapters, embeddings, etc.)
│   ├── processors/         # Preprocessing modules
│   ├── tasks/              # Task-specific logic (e.g., translation)
│   ├── training_setup/     # Training pipeline setup
│   ├── multimodalhugs_cli/ # Command-line interface for training/inference
│   └── utils/              # Helper functions
├── scripts/                # Utility scripts (e.g., documentation generation)
├── tests/                  # Unit tests
└── .github/                # GitHub actions and workflows
```

For a detailed breakdown of each directory, see [docs/README.md](docs/README.md).

## Contributing

All contributions—bug reports, feature requests, or pull requests—are welcome. Please see our [GitHub repository](https://github.com/GerrySant/multimodalhugs) to get involved.

## License

This project is licensed under the terms of the MIT License.

## Citing this Work

If you use MultimodalHugs in your research or applications, please cite:

```bibtex
@misc{multimodalhugs2024,
    title={MultimodalHugs: A Reproducibility-Driven Framework for Multimodal Machine Translation},
    author={Sant, Gerard and Moryossef, Amit and Jiang, Zifan and Escolano, Carlos},
    howpublished={\url{https://github.com/GerrySant/multimodalhugs}},
    year={2024}
}
```

