#!/usr/bin/env bash
# ----------------------------------------------------------
# Sample Script for Image2Text Translation (Generalized)
# ----------------------------------------------------------
# This script demonstrates a general approach to:
#  1. Setting up environment variables.
#  2. Preprocessing data.
#  3. Configuring and starting a multimodal translation run.
# ----------------------------------------------------------

# (Optional) Activate your conda environment
# source /path/to/anaconda/bin/activate <ENV_NAME>

# ----------------------------------------------------------
# 1. Define environment variables
# ----------------------------------------------------------
export MODEL_NAME="hebrew_image_to_text"
export REPO_PATH="/path/to/repositories"
export CONFIG_PATH="${REPO_PATH}/examples/multimodal_translation/image2text_translation/configs/example_config.yaml"
export OUTPUT_PATH="/path/to/experiments/output_directory"
export SOURCE_PROMPT="__vhe__"
export GENERATION_PROMPT="__en__"
export CUDA_VISIBLE_DEVICES=0
export EVAL_STEPS=250

# ----------------------------------------------------------
# 2. Preprocess Data
# ----------------------------------------------------------
python ${REPO_PATH}/multimodalhugs/examples/multimodal_translation/image2text_translation/example_scripts/hebrew_dataset_preprocessing_script.py \
    /path/to/data/dev/source.txt \
    /path/to/data/dev/target.txt \
    $SOURCE_PROMPT \
    $GENERATION_PROMPT \
    /path/to/data/dev/metadata.tsv

python ${REPO_PATH}/multimodalhugs/examples/multimodal_translation/image2text_translation/example_scripts/hebrew_dataset_preprocessing_script.py \
    /path/to/data/devtest/source.txt \
    /path/to/data/devtest/target.txt \
    $SOURCE_PROMPT \
    $GENERATION_PROMPT \
    /path/to/data/devtest/metadata.tsv

python ${REPO_PATH}/multimodalhugs/examples/multimodal_translation/image2text_translation/example_scripts/hebrew_dataset_preprocessing_script.py \
    /path/to/data/train/source.txt \
    /path/to/data/train/target.txt \
    $SOURCE_PROMPT \
    $GENERATION_PROMPT \
    /path/to/data/train/metadata.tsv

# ----------------------------------------------------------
# 3. Prepare Training Environment
# ----------------------------------------------------------
output=$(multimodalhugs-setup --modality "image2text" --config_path $CONFIG_PATH)

export MODEL_PATH=$(echo "$output" | grep 'MODEL_PATH' | cut -d '=' -f 2)
export PROCESSOR_PATH=$(echo "$output" | grep 'PROCESSOR_PATH' | cut -d '=' -f 2)
export DATA_PATH=$(echo "$output" | grep 'DATA_PATH' | cut -d '=' -f 2)

# Display environment variables for debugging
echo "MODEL_NAME: $MODEL_NAME"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH: $MODEL_PATH"
echo "PROCESSOR_PATH: $PROCESSOR_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "EVAL_STEPS: $EVAL_STEPS"

# ----------------------------------------------------------
# 4. Train the Model
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
    --eval_steps $EVAL_STEPS \
    --save_strategy "steps" \
    --save_steps $EVAL_STEPS \
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
    --lr_scheduler_type "inverse_sqrt" \
    --report_to none
