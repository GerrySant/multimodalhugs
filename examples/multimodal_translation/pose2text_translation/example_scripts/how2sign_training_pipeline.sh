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
# This command line sets up environment variables for the model, processor, etc
multimodalhugs-setup --modality "pose2text" --config_path $CONFIG_PATH

# ----------------------------------------------------------
# 4. Train the Model
# ----------------------------------------------------------
multimodalhugs-train \
    --task "translation" \
    --config-path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH