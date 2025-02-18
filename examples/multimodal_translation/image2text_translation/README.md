
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

##### Example Metadata File Format

Below is an example of how your metadata file should be structured. Each row represents one sample, and the columns correspond to the required fields:

| **source_signal**   | **source_prompt**         | **generation_prompt** | **output_text**                                                                   |
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
multimodalhugs-setup --modality "image2text" --config-path </path/to/example_config.yaml>
```
The script will automatically save the paths of each of the training actors created in their respective section of the configuration.

> **Note:** In this example, the model uses `m2m_100` as a pretrained backbone, along with its corresponding tokenizer. This can be seen in the configuration fields:  
> - `model.pretrained_backbone: facebook/m2m100_418M`  
> - `data.text_tokenizer_path: facebook/m2m100_418M`  
>   
> If you want to initialize a backbone from scratch, select the desired architecture in `model.backbone_name` (e.g., `m2m_100`), set `model.pretrained_backbone: null`, and define the necessary hyperparameters under `model.backbone_config`. For instance:
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
    --config-path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH \
    [--additional-arg <value> ...]  # Additional optional arguments can be specified here.
```
**Note:**  
The `--config-path` (Optional) parameter indicates that the training arguments defined in the configuration file will be used if they have not been specified on the command line.

>**Tip**: Adjust hyperparameters in the script or pass them via command line. MultimodalHugs integrates seamlessly with Hugging Face’s `Seq2SeqTrainer`, giving you flexibility for learning rates, batch sizes, evaluation intervals, etc.

## Full Pipeline Example

The script [`hebrew_training_pipeline.sh`](./example_scripts/hebrew_training_pipeline.sh) performs the entire pipeline, taking care of both the preprocessing of the dataset, the setup of the training modules and finally launches a training example. Simply run the following command:

```bash
bash hebrew_training_pipeline.sh
```


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
    ├── Arial.ttf                   # File needed for the creation of the images
    └── new_languages_hebrew.txt    # Contains a dictionary with the new tokens to be added to the tokenizer.
```
- `configs`: Contains YAML config files (e.g., model / data / training hyperparameters).
- `example_scripts`: Contains sample Python and bash scripts for preprocessing, setting up training, and launching experiments.
- `other`: Additional resources (e.g., a file listing new tokens).
