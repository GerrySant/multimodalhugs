
# Image2Text Translation (Hebrew Example)

This directory showcases how to prepare and train an **Image2Text** translation model using the **MultimodalHugs** framework with a Hebrew dataset as an example. Below, you will find step-by-step instructions on:

1. **Dataset Preparation**  
2. **Setting up the Training Environment**  
3. **Launching the Training Process**
4. **Evaluating the Trained Model**

> **Note**: These scripts rely on MultimodalHugs’ internal “autoclasses” (e.g., `Image2TextTranslationProcessor`, `MultiModalEmbedderModel`) and Hugging Face’s `Seq2SeqTrainer` to orchestrate dataset loading, tokenization, processing, and training.

---

## 1. Dataset Preparation

**Goal**: Convert raw text files into `.tsv` metadata suitable for Image2Text training.

- **Script**: [`hebrew_dataset_preprocessing_script.py`](./example_scripts/hebrew_dataset_preprocessing_script.py)  
- **Input**: Text files containing the source and target sentences, along with optional prompts.  
- **Output**: A `.tsv` file with standardized columns for training.

For each partition (train, val, test), run the script:
```bash
python hebrew_dataset_preprocessing_script.py "/path/to/source.txt" "/path/to/target.txt" "__vhe__" "__en__" "/path/to/output.tsv"
```

#### Metadata File Requirements

The `metadata.tsv` files for each partition must include the following fields:

- `signal`: The source text for the translation from which the images will be created / The path of the images to be uploaded (currently with support for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.npy`)
- `encoder_prompt`: A text string (e.g., `__vhe__`) that helps the model distinguish the source language or modality. Can be empty if not used.
- `decoder_prompt`: A text prompt appended during decoding to guide the model’s generation. Useful for specifying style or language; can be empty if not used.
- `output`: The target text for translation.

##### Example Metadata File Format

Below is an example of how your metadata file should be structured. Each row represents one sample, and the columns correspond to the required fields:

| **signal**   | **encoder_prompt**         | **decoder_prompt** | **output**                                                                   |
|-----------------------------------------------------------|---------------------------|-----------------------|-----------------------------------------------------------------------------------|
| `/path/to/your/image_1.jpg /path/to/your/image_2.jpg ... /path/to/your/image_N.jpg`         | `__vhe__`     |           `__en__`            | `No one was inside the apartment`                                                                               |
| `הטייס זוהה כמפקד הטייסת דילוקריט פאטאבי.`                    | `__vhe__`     |              `__en__`           | `The pilot was identified as Squadron Leader Dilokrit Pattavee.`                       |
| `בית הסוהר אבו-גרייב בעירק הוצת במהלך התפרעות.`                     | `__vhe__`    |             `__en__`            | `Iraq's Abu Ghraib prison has been set alight during a riot.`                |
| `/path/to/your/image_1.npy /path/to/your/image_2.npy ... /path/to/your/image_M.npy`                 | `__vhe__`    |           `__ar__`              | `Zayat was unhurt in the accident.`| 


## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, create or load the model, and save paths for easy retrieval.

- **Command Line**: `multimodalhugs-setup --modality "image2text"`
- **Input**: A config file (e.g., `configs/example_config.yaml`) specifying:
  - Model parameters
  - Tokenizer paths
  - Training paths (output directories, etc.)
  - Additional vocabulary files (e.g., `other/new_languages_hebrew.txt`)

Run the setup:

```bash
multimodalhugs-setup --modality "image2text" \
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



## 3. Launching the Training Process

**Goal**: Start the full Image2Text training routine using Hugging Face’s Trainer.

- **Process**:
  1. (Optional) Activate a virtual environment or conda environment.
  2. Define environment variables (e.g., `MODEL_NAME`, `MODEL_PATH`, `PROCESSOR_PATH`, `OUTPUT_PATH`).
  5. Run `multimodalhugs-train` with the desired arguments.

- **Training Command Lines:**:

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

The script [`hebrew_training_pipeline.sh`](./example_scripts/hebrew_training_pipeline.sh) performs the entire pipeline, taking care of both the preprocessing of the dataset, the setup of the training modules and finally launches a training example. Simply run the following command:

```bash
bash hebrew_training_pipeline.sh
```

## 4. Evaluating the Trained Model

**Goal**: Generate translations and evaluate model performance using evaluation metrics (e.g., sacreBLEU).

After training, you can evaluate your model with the `multimodalhugs-generate` command. This command loads your trained checkpoint, processor, and dataset to generate translations and compute evaluation metrics.

Below is a general bash script outline that you can adapt for evaluation. Replace the placeholder paths with your actual directories:

```bash
multimodalhugs-generate \
    --task "translation" \                       # Identifier of the task to be evaluated.
    --metric_name "sacrebleu" \                  # Evaluation metric identifier (e.g., 'sacrebleu').
    --output_dir $OUTPUT_PATH \                  # Directory to store generation outputs and metrics.
    --model_name_or_path $CKPT_PATH \            # Trained model checkpoint directory.
    --processor_name_or_path $PROCESSOR_PATH \   # Directory to the processor created via multimodalhugs-setup.
    --dataset_dir $DATA_PATH \                   # Directory of the dataset created by multimodalhugs-setup.
    --config_path $CONFIG_PATH                   # (Optional) Additional configuration parameters.
```

**Key arguments explained:**
- `--task "translation"`: Specifies the evaluation task.
- `--metric_name "sacrebleu"`: Sets the evaluation metric to sacreBLEU (you may choose another metric as needed).
- `--output_dir`: Directory where evaluation outputs (generated translations and metrics) will be saved.
- `--model_name_or_path`: Path to your trained model checkpoint.
- `--processor_name_or_path`: Path to the processor created during the setup phase.
- `--dataset_dir`: Directory of your preprocessed dataset.
- `--config_path`: (Optional) Path to a YAML configuration file containing additional parameters.
  
>**Tip**: As in training, the parameters  `model_name_or_path`, `processor_name_or_path` and `dataset_dir` can also be specified within the config.

This evaluation process enables you to assess the quality of your model's translations and compare them against reference texts.

## Directory Overview
```kotlin
image2text_translation
├── README.md                  # Current documentation
├── configs
│   └── example_config.yaml    # Example config template
├── example_scripts
│   ├── hebrew_dataset_preprocessing_script.py
│   └── hebrew_training_pipeline.sh
└── other
    └── Arial.ttf                   # File needed for the creation of the images
```
- `configs`: Contains YAML config files (e.g., model / data / training hyperparameters).
- `example_scripts`: Contains sample Python and bash scripts for preprocessing, setting up training, and launching experiments.
- `other`: Additional resources (e.g., file containig the font used).
