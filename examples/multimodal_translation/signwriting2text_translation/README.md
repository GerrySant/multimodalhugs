
# SignWriting2Text Translation Example

This directory provides an example of preparing and training a **SignWriting2Text** translation model using the **MultimodalHugs** framework. The workflow includes the following steps:

1. **Dataset Preparation**
2. **Setting Up the Training Environment**
3. **Launching the Training Process**

> **Note**: This example uses MultimodalHugs' specialized classes (e.g., `SignWritingProcessor`, `MultiModalEmbedderModel`) alongside Hugging Face’s `Seq2SeqTrainer` to manage dataset loading, tokenization, preprocessing, and training.

---

## 1. Dataset Preparation

**Goal**: Convert raw CSV files into `.tsv` metadata files for SignWriting2Text translation.

- **Script**: [`signbankplus_dataset_preprocessing_script.py`](./example_scripts/signbankplus_dataset_preprocessing_script.py)
- **Input**: A CSV file containing fields like `source`, `target`, `src_lang`, and `tgt_lang`.
- **Output**: A `.tsv` file formatted for training.

Run the script for each data partition (train, dev, test):
```bash
python signbankplus_dataset_preprocessing_script.py /path/to/input.csv /path/to/output.tsv
```

### Metadata File Requirements

The `metadata.tsv` files must contain the following fields:

- `input`: The SignWriting source sequence.
- `source_prompt`: A text string (e.g., `__signwriting__ __en__`) to guide modality and language processing.
- `generation_prompt`: A prompt for the target language.
- `output_text`: The corresponding text translation.

These fields ensure compatibility with the **SignWriting2Text** processing pipeline.

---

## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, preprocessors, and models, and save their paths for further usage.

- **Script**: [`signbankplus_training_setup.py`](./example_scripts/signbankplus_training_setup.py)
- **Input**: A configuration file (e.g., `configs/example_config.yaml`) specifying:
  - Model parameters
  - Tokenizer paths
  - Dataset paths
  - Additional vocabulary (e.g., `other/new_languages_sign_bank_plus.txt`)
- **Output**:
  - `MODEL_PATH`: Directory with the trained model.
  - `PROCESSOR_PATH`: Directory with the saved processor.
  - `DATA_PATH`: Directory where the processed dataset is stored.

Run the setup script:

```bash
python signbankplus_training_setup.py --config_path /path/to/signwriting_config.yaml
```

The script outputs environment variables (`MODEL_PATH`, `PROCESSOR_PATH`, `DATA_PATH`) for downstream usage.

---

## 3. Launching the Training Process

**Goal**: Train the **SignWriting2Text** model using the prepared data and configurations.

- **Script**: [`signbankplus_training.sh`](./example_scripts/signbankplus_training.sh)
- **Process**:
  1. (Optional) Activate your conda or virtual environment.
  2. Define environment variables (`MODEL_NAME`, `CONFIG_PATH`, `OUTPUT_PATH`).
  3. Preprocess datasets using the `signbankplus_dataset_preprocessing_script.py` script.
  4. Run `signbankplus_training_setup.py` to initialize paths and settings.
  5. Start training with `run_translation.py`.

Run the script:

```bash
bash signbankplus_training.sh
```

### Hyperparameter Tuning

Modify the `signbankplus_training.sh` script or pass arguments via the command line to adjust hyperparameters such as batch size, learning rate, and evaluation steps.

---

## Directory Overview

```plaintext
signwriting2text_translation
├── README.md                  # Documentation (current file)
├── configs
│   └── example_config.yaml    # Example configuration file
├── example_scripts
│   ├── signbankplus_dataset_preprocessing_script.py
│   ├── signbankplus_training.sh
│   └── signbankplus_training_setup.py
└── other
    └── new_languages_sign_bank_plus.txt  # Additional tokens for the tokenizer
```

### Directory Details
- **configs**: Contains YAML configuration files for model, dataset, and training parameters.
- **example_scripts**: Includes scripts for dataset preprocessing, training setup, and execution.
- **other**: Additional resources such as vocabulary files for new tokens.
