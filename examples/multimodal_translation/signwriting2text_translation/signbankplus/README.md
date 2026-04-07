
# SignWriting2Text Translation Example

This directory provides an example of preparing and training a **SignWriting2Text** translation model using the **MultimodalHugs** framework. The workflow includes the following steps:

1. **Dataset Preparation**
2. **Setting Up the Training Environment**
3. **Launching the Training Process**
4. **Evaluating the Trained Model**

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

- `signal`: The SignWriting source sequence.
- `encoder_prompt`: A text string (e.g., `__signwriting__ __en__`) to guide modality and language processing.
- `decoder_prompt`: A prompt for the target language.
- `output`: The corresponding text translation.

These fields ensure compatibility with the **SignWriting2Text** processing pipeline.

##### Example Metadata File Format

Below is an example of how your metadata file should be structured. Each row represents one sample, and the columns correspond to the required fields:

| **signal**   | **encoder_prompt**         | **decoder_prompt** | **output**                                                                   |
|-----------------------------------------------------------|---------------------------|-----------------------|-----------------------------------------------------------------------------------|
| `M526x565S30004482x483S20710484x522S15d52499x546`           | `__ncs__`     |           `__es__`            | `dificil`                                                                               |
| `M521x530S1f540506x489S1f502481x471S20e00484x499S22f14479x516`                    | `__ase__`     |              `__en__`           | `Every book has been stolen.`                       |
| `AS1ce20S14a20S14720M530x515S1ce20469x485S14a20494x500S14720516x493`                     | `__ase__`    |             `__en__`            | `February`                |
| `M528x518S2ff00482x483S16d40510x462S16d48472x462S22b00512x427S22b10472x427S30a00482x483`                 | `__jos__`    |           `__ar__`              | `قبعة طاقية غطاء الراس`| 

---

## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, preprocessors, and models, and save their paths for further usage.

- **Command Line**: `multimodalhugs-setup --modality "signwriting2text"`
- **Input**: A configuration file (e.g., `configs/example_config.yaml`) specifying:
  - Model parameters
  - Tokenizer paths
  - Dataset paths
  - Additional vocabulary (e.g., `other/new_languages_sign_bank_plus.txt`)

Run the setup script:

```bash
multimodalhugs-setup --modality "signwriting2text" \
  --config_path </path/to/example_config.yaml> \
  --output_dir </path/to/your/output_directory>
```
The script will automatically create the training actors at `</path/to/your/output_directory>/setup` and save their paths (needed for the `multimodalhugs-train`) at `</path/to/your/output_directory>/setup/actors_paths.yaml`.

> **Note:** In this example, the model uses `m2m_100` as a pretrained backbone, along with its corresponding tokenizer. This can be seen in the configuration fields:  
> - `model.pretrained_backbone: facebook/m2m100_418M`  
> - `data.text_tokenizer_path: facebook/m2m100_418M`  
>   
> If you want to initialize a backbone from scratch, select the desired architecture in `model.backbone_type` (e.g., `m2m_100`), set `model.pretrained_backbone: null`, and define the necessary hyperparameters under `model.backbone_config`. For instance:
>
> ```yaml
> backbone_config:
>   vocab_size: 384
>   bos_token_id: 0
>   eos_token_id: 1
>   pad_token_id: 0
>   decoder_start_token_id: 0
>   d_model: 256  
>   encoder_layers: 6 
>   encoder_attention_heads: 1
>   decoder_layers: 6  
>   decoder_attention_heads: 1
>   decoder_ffn_dim: 64
>   encoder_ffn_dim: 64
>   (...)
> ```
>
> If you plan to use a custom tokenizer, make sure to train a tokenizer compatible with your backbone and specify its path in the configuration:  
> `data.text_tokenizer_path: path/to/local/tokenizer/`

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
export CONFIG_PATH="/path/to/config.yaml"
export OUTPUT_PATH="/path/to/your/output_directory" # Directory to store model checkpoints & results.
export WANDB_PROJECT="<my_prpject_name>" # To specify the WANDB project

# ----------------------------------------------------------
# 2. Train the Model
# ----------------------------------------------------------
multimodalhugs-train \
    --task "translation" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH \
    [--additional-arg <value> ...]  # Additional optional arguments can be specified here.
```
**Note:**  
The `--config_path` (Optional) parameter indicates that the training arguments defined in the configuration file will be used if they have not been specified on the command line.

>**Tip**: Adjust hyperparameters in the script or pass them via command line. MultimodalHugs integrates seamlessly with Hugging Face’s `Seq2SeqTrainer`, giving you flexibility for learning rates, batch sizes, evaluation intervals, etc.


## Full Training Pipeline Example

The script [`signbankplus_training_pipeline.sh`](./example_scripts/signbankplus_training_pipeline.sh) performs the entire pipeline, taking care of both the preprocessing of the dataset, the setup of the training modules and finally launches a training example. Simply run the following command:

```bash
bash signbankplus_training_pipeline.sh
```

## 4. Evaluating the Trained Model

**Goal**: Generate translations and evaluate model performance using evaluation metrics (e.g., sacreBLEU).

After training, you can evaluate your model with the `multimodalhugs-generate` command. This command loads your trained checkpoint, processor, and dataset to generate translations and compute evaluation metrics.

Below is a general bash script outline that you can adapt for evaluation. Replace the placeholder paths with your actual directories:

```bash
multimodalhugs-generate \
    --task "translation" \                       # Identifier of the task to be evaluated.
    --metric_name "sacrebleu" \                  # Evaluation metric identifier (e.g., 'sacrebleu').
    --generate_output_dir $OUTPUT_PATH \                  # Directory to store generation outputs and metrics.
    --model_name_or_path $CKPT_PATH \            # Trained model checkpoint directory.
    --processor_name_or_path $PROCESSOR_PATH \   # Directory to the processor created via multimodalhugs-setup.
    --dataset_dir $DATA_PATH \                   # Directory of the dataset created by multimodalhugs-setup.
    --config_path $CONFIG_PATH                   # (Optional) Additional configuration parameters.
```

**Key arguments explained:**
- `--task "translation"`: Specifies the evaluation task.
- `--metric_name "sacrebleu"`: Sets the evaluation metric to sacreBLEU (you may choose another metric as needed).
- `--generate_output_dir`: Directory where evaluation outputs (generated translations and metrics) will be saved.
- `--model_name_or_path`: Path to your trained model checkpoint.
- `--processor_name_or_path`: Path to the processor created during the setup phase.
- `--dataset_dir`: Directory of your preprocessed dataset.
- `--config_path`: (Optional) Path to a YAML configuration file containing additional parameters.
  
>**Tip**: As in training, the parameters  `model_name_or_path`, `processor_name_or_path` and `dataset_dir` can also be specified within the config.

This evaluation process enables you to assess the quality of your model's translations and compare them against reference texts.


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
│   └── signbankplus_training_pipeline.sh
└── other
    └── new_languages_sign_bank_plus.txt  # Additional tokens for the tokenizer
```

### Directory Details
- **configs**: Contains YAML configuration files for model, dataset, and training parameters.
- **example_scripts**: Includes scripts for dataset preprocessing, training setup, and execution.
- **other**: Additional resources such as vocabulary files for new tokens.
