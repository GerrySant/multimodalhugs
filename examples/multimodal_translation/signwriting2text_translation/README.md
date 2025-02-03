
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

- `source_signal`: The SignWriting source sequence.
- `source_prompt`: A text string (e.g., `__signwriting__ __en__`) to guide modality and language processing.
- `generation_prompt`: A prompt for the target language.
- `output_text`: The corresponding text translation.

These fields ensure compatibility with the **SignWriting2Text** processing pipeline.

##### Example Metadata File Format

Below is an example of how your metadata file should be structured. Each row represents one sample, and the columns correspond to the required fields:

| **source_signal**   | **source_prompt**         | **generation_prompt** | **output_text**                                                                   |
|-----------------------------------------------------------|---------------------------|-----------------------|-----------------------------------------------------------------------------------|
| `M526x565S30004482x483S20710484x522S15d52499x546`           | `__ncs__`     |           `__es__`            | `dificil`                                                                               |
| `M521x530S1f540506x489S1f502481x471S20e00484x499S22f14479x516`                    | `__ase__`     |              `__en__`           | `Every book has been stolen.`                       |
| `AS1ce20S14a20S14720M530x515S1ce20469x485S14a20494x500S14720516x493`                     | `__ase__`    |             `__en__`            | `February`                |
| `M528x518S2ff00482x483S16d40510x462S16d48472x462S22b00512x427S22b10472x427S30a00482x483`                 | `__jos__`    |           `__ar__`              | `قبعة طاقية غطاء الراس`| 

---

## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, preprocessors, and models, and save their paths for further usage.

- **Script**: [`signwriting2text_training_setup.py`](./example_scripts/signwriting2text_training_setup.py)
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
python signwriting2text_training_setup.py --config_path /path/to/signwriting_config.yaml
```

The script outputs environment variables (`MODEL_PATH`, `PROCESSOR_PATH`, `DATA_PATH`) for downstream usage.

---

## 3. Launching the Training Process

**Goal**: Train the **SignWriting2Text** model using the prepared data and configurations.

- **Process**:
  1. (Optional) Activate your conda or virtual environment.
  2. Define environment variables (`MODEL_NAME`, `MODEL_PATH`, `OUTPUT_PATH`).
  3. Start training with `multimodalhugs-train`.

**Training Command Lines:**:

```bash
# ----------------------------------------------------------
# 1. Specify global variables
# ----------------------------------------------------------
export MODEL_NAME="signwriting2text_example"
export OUTPUT_PATH="/path/to/your/output_directory"
export MODEL_PATH="/obtained/by/signbankplus_dataset_preprocessing_script.py"
export PROCESSOR_PATH="/obtained/by/signbankplus_dataset_preprocessing_script.py"
export DATA_PATH="/obtained/by/signbankplus_dataset_preprocessing_script.py"

# ----------------------------------------------------------
# 2. Train the Model
# ----------------------------------------------------------
multimodalhugs-train \
    --task "translation" \
    --model_name_or_path $MODEL_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --run_name $MODEL_NAME \
    --dataset_dir $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --do_train True \
    --do_eval True \
    --logging_steps 100 \
    --remove_unused_columns False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model 'bleu' \
    --overwrite_output_dir \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --warmup_steps 20000 \
    --max_steps 200000 \
    --predict_with_generate True \
    --lr_scheduler_type "inverse_sqrt"
```

## Full Pipeline Example

The script [`signbankplus_training_pipeline.sh`](./example_scripts/signbankplus_training_pipeline.sh) performs the entire pipeline, taking care of both the preprocessing of the dataset, the setup of the training modules and finally launches a training example. Simply run the following command:

```bash
bash signbankplus_training_pipeline.sh
```

### Hyperparameter Tuning

Modify the `signbankplus_training_pipeline.sh` script or pass arguments via the command line to adjust hyperparameters such as batch size, learning rate, and evaluation steps.

---

## Directory Overview

```plaintext
signwriting2text_translation
├── README.md                  # Documentation (current file)
├── configs
│   └── example_config.yaml    # Example configuration file
├── example_scripts
│   ├── signbankplus_dataset_preprocessing_script.py
│   ├── signbankplus_training_pipeline.sh
│   └── signwriting2text_training_setup.py
└── other
    └── new_languages_sign_bank_plus.txt  # Additional tokens for the tokenizer
```

### Directory Details
- **configs**: Contains YAML configuration files for model, dataset, and training parameters.
- **example_scripts**: Includes scripts for dataset preprocessing, training setup, and execution.
- **other**: Additional resources such as vocabulary files for new tokens.
