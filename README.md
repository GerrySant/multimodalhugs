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
Explore the [examples/multimodal_translation/](/examples/multimodal_translation/) directory for an end-to-end workflow demonstrating how to:

1. **Preprocess Data**: Convert raw data into `.tsv` format with columns for prompts, languages, target labels, etc.
2. **Configure Training**: Tune model hyperparameters via YAML or Python script.
3. **Train & Evaluate**: Utilize the included training scripts and Hugging Face's Trainer for effortless experimentation.
4. **Extend & Adapt**: Incorporate custom datasets, tokenizers, or specialized processing modules.

>**Note**: Each example folder (e.g., Image2text_translation, pose2text_translation) contains its own detailed documentation. Refer there for more specifics.

## Directory Overview
```kotlin
multimodalhugs
├── examples
│   └── multimodal_translation
│       ├── image2text_translation
│       ├── pose2text_translation
│       └── signwriting2text_translation
├── multimodalhugs
│   ├── custom_datasets
│   ├── data
│   ├── models
│   ├── modules
│   ├── multimodalhugs_cli
│   ├── processors
│   ├── tasks
│   ├── training_setup
│   └── utils
└── tests
```

- `examples/`: Contains ready-to-run demos for various multimodal tasks.
- `multimodalhugs/`: Core library code (datasets, models, modules, etc.).
- `tests/`: Automated tests to ensure code integrity.

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
