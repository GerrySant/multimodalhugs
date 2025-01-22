
# Image2Text Translation (Hebrew Example)

This directory showcases how to prepare and train an **Image2Text** translation model using the **MultimodalHugs** framework with a Hebrew dataset as an example. Below, you will find step-by-step instructions on:

1. **Dataset Preparation**  
2. **Setting up the Training Environment**  
3. **Launching the Training Process**

> **Note**: These scripts rely on MultimodalHugs’ internal “autoclasses” (e.g., `Image2TextTranslationProcessor`, `MultiModalEmbedderModel`) and Hugging Face’s `Seq2SeqTrainer` to orchestrate dataset loading, tokenization, processing, and training.

---

## 1. Dataset Preparation

**Goal**: Convert raw text files into `.tsv` metadata suitable for Image2Text training.

- **Script**: [`hebrew_dataset_preprocessing_script.py`](./example_scripts/hebrew_dataset_preprocessing_script.py)  
- **Input**: Text files containing the source and target sentences, along with optional prompts.  
- **Output**: A `.tsv` file with standardized columns for training.

For each partition (train, val, test), run the script:
```bash
python hebrew_dataset_preprocessing_script.py /path/to/source.txt /path/to/target.txt __vhe__ __en__ /path/to/output.tsv
```

#### Metadata File Requirements

The `metadata.tsv` files for each partition must include the following fields:

- `source_signal`: The source text for the translation from which the images will be created / The path of the images to be uploaded (currently with support for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.npy`)
- `source_prompt`: A text string (e.g., `__vhe__`) that helps the model distinguish the source language or modality. Can be empty if not used.
- `generation_prompt`: A text prompt appended during decoding to guide the model’s generation. Useful for specifying style or language; can be empty if not used.
- `output_text`: The target text for translation.

## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, create or load the model, and save paths for easy retrieval.

- **Script**: `hebrew_training_setup.py`
- **Input**: A config file (e.g., `configs/example_config.yaml`) specifying:
  - Model parameters
  - Tokenizer paths
  - Training paths (output directories, etc.)
  - Additional vocabulary files (e.g., `other/new_languages_hebrew.txt`)
- **Output**:
  1. `MODEL_PATH`: Directory with the saved model.
  2. `PROCESSOR_PATH`: Directory with the saved processor.
  3. `DATA_PATH`: Directory where the processed dataset is stored.

Run the setup:

```bash
python hebrew_training_setup.py --config_path /path/to/example_config.yaml
```
The script will print environment variables (`MODEL_PATH`, `PROCESSOR_PATH`, `DATA_PATH`) that you can export for downstream usage.

## 3. Launching the Training Process

**Goal**: Start the full Image2Text training routine using Hugging Face’s Trainer.

- **Script**: [`hebrew_training.sh`](./example_scripts/hebrew_training.sh)
- **Process**:
  1. (Optional) Activate a virtual environment or conda environment.
  2. Define environment variables (e.g., `MODEL_NAME`, `REPO_PATH`, `CONFIG_PATH`, `OUTPUT_PATH`).
  3. Preprocess the dataset (calling `hebrew_dataset_preprocessing_script.py`).
  4. Run `hebrew_training_setup.py` to configure and retrieve paths.
  5. Invoke `run_translation.py` with the correct arguments.

Example usage:

```bash
bash hebrew_training.sh
```

>**Tip**: Adjust hyperparameters in the script or pass them via command line. MultimodalHugs integrates seamlessly with Hugging Face’s `Seq2SeqTrainer`, giving you flexibility for learning rates, batch sizes, evaluation intervals, etc.

## Directory Overview
```kotlin
image2text_translation
├── README.md                  # Current documentation
├── configs
│   └── example_config.yaml    # Example config template
├── example_scripts
│   ├── hebrew_dataset_preprocessing_script.py
│   ├── hebrew_training.sh
│   └── hebrew_training_setup.py
└── other
    ├── Arial.ttf                   # File needed for the creation of the images
    └── new_languages_hebrew.txt    # Contains a dictionary with the new tokens to be added to the tokenizer.
```
- `configs`: Contains YAML config files (e.g., model / data / training hyperparameters).
- `example_scripts`: Contains sample Python and bash scripts for preprocessing, setting up training, and launching experiments.
- `other`: Additional resources (e.g., a file listing new tokens).
