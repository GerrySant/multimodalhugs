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

- `source_signal`: Path to the input pose.
- `source_start`: Start timestamp (commonly in milliseconds) of the input segment. Can be left empty or `0` if not required by the setup.
- `source_end`: End timestamp (commonly in milliseconds) of the input segment. Can be left empty or `0` if not required by the setup.
- `source_prompt`: A text string (e.g., `__pose__ __en__`) that helps the model distinguish the modality or language. Can be empty if not used.
- `generation_prompt`: A text prompt appended during decoding to guide the model’s generation. Useful for specifying style or language; can be empty if not used.
- `output_text`: Target text for translation.
  
If `source_start` and `source_end` are not required (when all the frames are used), must be set as `0`. If `source_prompt` and `generation_prompt` are nout needed, can be let empty.

##### Example Metadata File Format

Below is an example of how your metadata file should be structured. Each row represents one sample, and the columns correspond to the required fields:

| **source_signal**                                         | **source_start** | **source_end** | **source_prompt**         | **generation_prompt** | **output_text**                                                                   |
|-----------------------------------------------------------|------------------|----------------|---------------------------|-----------------------|-----------------------------------------------------------------------------------|
| `/path/to/pose1.pose`               | `0`                | `0`              | `__slt__ __asl__`     |           `__en__`            | `Hi!`                                                                               |
| `/path/to/pose2.pose`               | `24`             | `385`              | `__slt__ __asl__`     |              `__en__`           | `The aileron is controlled by lateral movement of the stick.`                       |
| `/path/to/pose2.pose`               | `404`                | `514`              | `__slt__ __asl__`    |             `__en__`            | `By moving the stick, the angle of attack is adjusted for that wing.`                |
| `/path/to/pose3.pose`               | `63`                | `88`            | `__slt__ __asl__`    |           `__en__`              | `The elevator adjusts the airplane's angle of attack.`                             |



## 2. Setting Up the Training Environment

**Goal**: Initialize tokenizers, create or load the model, and save paths for easy retrieval.

- **Command Line**: `multimodalhugs-setup --modality "pose2text"`
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
multimodalhugs-setup --modality "pose2text" --config_path </path/to/example_config.yaml>
```
The script will print environment variables (`MODEL_PATH`, `PROCESSOR_PATH`, `DATA_PATH`) that you can export for downstream usage.

> **Note:** In this example, the model uses `m2m_100` as a pretrained backbone, along with its corresponding tokenizer. This can be seen in the configuration fields:  
> - `model.pretrained_backbone: facebook/m2m100_418M`  
> - `data.text_tokenizer_path: facebook/m2m100_418M`  
>   
> If you want to initialize a backbone from scratch, select the desired architecture in `model.backbone_name` (e.g., `m2m_100`), set `model.pretrained_backbone: null`, and define the necessary hyperparameters under `model.backbone_cfg`. For instance:
>
> ```yaml
> backbone_cfg:
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
**Goal**: Start the full Pose2Text training routine using Hugging Face’s Trainer.

- **Process**:
  1. (Optional) Activate a virtual environment or conda environment.
  2. Define environment variables (e.g., `MODEL_NAME`, `MODEL_PATH`, `PROCESSOR_PATH`, `OUTPUT_PATH`, etc).
  3. Run `multimodalhugs-train` with the desired arguments.

- **Training Command Lines:**:

```bash
# ----------------------------------------------------------
# 1. Specify global variables
# ----------------------------------------------------------
export MODEL_NAME="pose2text_example"
export OUTPUT_PATH="/path/to/your/output_directory"
export MODEL_PATH="/obtained/by/pose2text_training_setup.py"
export PROCESSOR_PATH="/obtained/by/pose2text_training_setup.py"
export DATA_PATH="/obtained/by/pose2text_training_setup.py"

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
    --fp16 \
    --label_smoothing_factor 0.1 \
    --remove_unused_columns False \
    --per_device_train_batch_size 8 \
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
    --predict_with_generate True
```
>**Tip**: Adjust hyperparameters in the script or pass them via command line. MultimodalHugs integrates seamlessly with Hugging Face’s `Seq2SeqTrainer`, giving you flexibility for learning rates, batch sizes, evaluation intervals, etc.


## Full Pipeline Example

The script [how2sign_training_pipeline.sh](https://github.com/GerrySant/multimodalhugs/blob/master/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_training_pipeline.py) performs the entire pipeline, taking care of both the preprocessing of the dataset, the setup of the training modules and finally launches a training example. Simply run the following command:

```bash
bash how2sign_training_pipeline.sh
```

## Directory Overview
```kotlin
pose2text_translation
├── README.md                  # Current documentation
├── configs
│   └── example_config.yaml    # Example config template
├── example_scripts
│   ├── how2sign_dataset_preprocessing_script.py
│   └── how2sign_training_pipeline.sh
└── other
    └── new_languages_how2sign.txt  # Optional: new tokens for tokenizer
```
- `configs`: Contains YAML config files (e.g., model / data / training hyperparameters).
- `example_scripts`: Contains sample Python and bash scripts for preprocessing, setting up training, and launching experiments.
- `other`: Additional resources (e.g., a file listing new tokens).
