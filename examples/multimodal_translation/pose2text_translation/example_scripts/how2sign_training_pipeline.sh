#!/usr/bin/env bash
# ----------------------------------------------------------
# Sample Script for Pose2Text Translation (Generalized)
# ----------------------------------------------------------
# This script demonstrates a more general approach to:
#  1. Setting up environment variables.
#  2. Preprocessing data.
#  3. Configuring and starting a multimodal translation run.
# ----------------------------------------------------------

# (Optional) Activate your conda environment
# source /path/to/your/anaconda/bin/activate <YOUR_ENV_NAME>

# ----------------------------------------------------------
# 1. Define environment variables
# ----------------------------------------------------------
export MODEL_NAME="pose2text_example"
export REPO_PATH="/path/to/your/multimodalhugs"
export CONFIG_PATH="/path/to/pose2text_config.yaml"
export OUTPUT_PATH="/path/to/your/output_directory"
export CUDA_VISIBLE_DEVICES=0

# ----------------------------------------------------------
# 2. Preprocess Data
# ----------------------------------------------------------
python ${REPO_PATH}/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_dataset_preprocessing_script.py \
    /path/to/how2sign_test.csv \
    /path/to/test_pose_files_directory \
    /path/to/how2sign_test_processed.tsv

python ${REPO_PATH}/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_dataset_preprocessing_script.py \
    /path/to/how2sign_val.csv \
    /path/to/val_pose_files_directory \
    /path/to/how2sign_val_processed.tsv

python ${REPO_PATH}/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_dataset_preprocessing_script.py \
    /path/to/how2sign_train.csv \
    /path/to/train_pose_files_directory \
    /path/to/how2sign_train_processed.tsv


# ----------------------------------------------------------
# 3. Prepare Training Environment
# ----------------------------------------------------------
# This Python script sets up environment variables for the model, processor, etc.
output=$(python ${REPO_PATH}/examples/multimodal_translation/pose2text_translation/example_scripts/pose2text_training_setup.py \
    --config_path $CONFIG_PATH)

# Extract environment variables from the Python script’s output
export MODEL_PATH=$(echo "$output" | grep 'MODEL_PATH' | cut -d '=' -f 2)
export PROCESSOR_PATH=$(echo "$output" | grep 'PROCESSOR_PATH' | cut -d '=' -f 2)
export DATA_PATH=$(echo "$output" | grep 'DATA_PATH' | cut -d '=' -f 2)

# Display the environment variables (for debugging)
echo "MODEL_NAME: $MODEL_NAME"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH: $MODEL_PATH"
echo "PROCESSOR_PATH: $PROCESSOR_PATH"
echo "DATA_PATH: $DATA_PATH"

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
    --predict_with_generate True \
    --lr_scheduler_type "inverse_sqrt"
    