# Pose2Text Translation (How2Sign Example)

This directory showcases how to prepare and train a **Pose2Text** translation model using the **MultimodalHugs** framework with the **How2Sign** dataset as an example. Below, you will find step-by-step instructions on:

1. **Dataset Preparation**  
2. **Setting up the Training Environment**  
3. **Launching the Training Process**

> **Note**: These scripts rely on MultimodalHugs’ internal “autoclasses” (e.g., `Pose2TextTranslationProcessor`, `MultiModalEmbedderModel`) and Hugging Face’s `Seq2SeqTrainer` to orchestrate dataset loading, tokenization, processing, and training.

---

## 1. Dataset Preparation

**Goal**: Convert raw CSV files into `.tsv` metadata suitable for Pose2Text training.

- **Script**: [`how2sign_dataset_preprocessing_script.py`](./example_scripts/how2sign_dataset_preprocessing_script.py)  
- **Input**: A CSV file containing fields like `VIDEO_NAME`, `START`, `END`, `SENTENCE`, `SENTENCE_NAME`.  
- **Output**: A `.tsv` file with standardized columns for training.

For each partition (train, val, test), run the script:
```bash
python how2sign_dataset_preprocessing_script.py /path/to/input.csv /path/to/output.tsv
```
#### Metadata File Requirements

The `metadata.tsv` files for each partition must include the following fields:

- `input`: Path to the input pose.
- `source_start`: Start timestamp (commonly in milliseconds) of the input segment. Can be left empty or `0` if not required by the setup.
- `source_end`: End timestamp (commonly in milliseconds) of the input segment. Can be left empty or `0` if not required by the setup.
- `input_text`: (optional) In case another text input is needed. (e.g., Siamese networks)
- `source_prompt`: A text string (e.g., `__pose__ __en__`) that helps the model distinguish the modality or language. Can be empty if not used.
- `generation_prompt`: A text prompt appended during decoding to guide the model’s generation. Useful for specifying style or language; can be empty if not used.
- `output_text`: Target text for translation.
  
If `source_start` and `source_end` are not required, must be set as `0`. If `source_prompt` and `generation_prompt` are nout needed, can be let empty.

## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, create or load the model, and save paths for easy retrieval.

- **Script**: how2sign_training_setup.py
- **Input**: A config file (e.g., `configs/example_config.yaml`) specifying:
  - Model parameters
  - Tokenizer paths
  - Training paths (output directories, etc.)
  - Additional vocabulary files (e.g., `other/new_languages_how2sign.txt`)
- **Output**:
  1. `MODEL_PATH`: Directory with the saved model.
  2. `PROCESSOR_PATH`: Directory with the saved processor.
  3. `DATA_PATH`: Directory where the processed dataset is stored.
   
Run the setup:

```bash
python how2sign_training_setup.py --config_path /path/to/pose2text_config.yaml
```
The script will print environment variables (`MODEL_PATH`, `PROCESSOR_PATH`, `DATA_PATH`) that you can export for downstream usage.



## 3. Launching the Training Process
**Goal**: Start the full Pose2Text training routine using Hugging Face’s Trainer.

- **Script**: [how2sign_training.sh](https://github.com/GerrySant/multimodalhugs/blob/master/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_training_setup.py)
- **Process**:
  1. (Optional) Activate a virtual environment or conda environment.
  2. Define environment variables (e.g., `MODEL_NAME`, `REPO_PATH`, `CONFIG_PATH`, `OUTPUT_PATH`).
  3. Preprocess the dataset (calling how2sign_dataset_preprocessing_script.py).
  4. Run `how2sign_training_setup.py` to configure and retrieve paths.
  5. Invoke `run_translation.py` with the correct arguments.


Example usage:

```bash
bash how2sign_training.sh
```
>**Tip**: Adjust hyperparameters in the script or pass them via command line. MultimodalHugs integrates seamlessly with Hugging Face’s `Seq2SeqTrainer`, giving you flexibility for learning rates, batch sizes, evaluation intervals, etc.

## Directory Overview
```kotlin
pose2text_translation
├── README.md                  # Current documentation
├── configs
│   └── example_config.yaml    # Example config template
├── example_scripts
│   ├── how2sign_dataset_preprocessing_script.py
│   ├── how2sign_training.sh
│   └── how2sign_training_setup.py
└── other
    └── new_languages_how2sign.txt  # Optional: new tokens for tokenizer
```
- `configs`: Contains YAML config files (e.g., model / data / training hyperparameters).
- `example_scripts`: Contains sample Python and bash scripts for preprocessing, setting up training, and launching experiments.
- `other`: Additional resources (e.g., a file listing new tokens).
