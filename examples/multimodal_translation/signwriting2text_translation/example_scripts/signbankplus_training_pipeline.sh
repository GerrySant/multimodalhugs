#!/usr/bin/env bash
# ----------------------------------------------------------
# Sample Script for Signwriting2Text Translation (Generalized)
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

export MODEL_NAME="src_prompt_signwriting"
export REPO_PATH="/path/to/your/repository"

export CONFIG_PATH="${REPO_PATH}/examples/multimodal_translation/signwriting2text_translation/configs/example_config.yaml/path/to/your/configs/signwriting_src_prompt.yaml"
export OUTPUT_PATH="/path/to/your/experiments/output_directory"
export CUDA_VISIBLE_DEVICES=0
export EVAL_STEPS=250

# ----------------------------------------------------------
# 2. Preprocess Data
# ----------------------------------------------------------
python ${REPO_PATH}/examples/multimodal_translation/signwriting2text_translation/example_scripts/signbankplus_dataset_preprocessing_script.py \
    /path/to/your/data/parallel/cleaned/dev.csv \
    /path/to/your/data/parallel/cleaned/dev_processed.tsv

python ${REPO_PATH}/examples/multimodal_translation/signwriting2text_translation/example_scripts/signbankplus_dataset_preprocessing_script.py \
    /path/to/your/data/parallel/cleaned/train_toy.csv \
    /path/to/your/data/parallel/cleaned/train_processed.tsv

python ${REPO_PATH}/examples/multimodal_translation/signwriting2text_translation/example_scripts/signbankplus_dataset_preprocessing_script.py \
    /path/to/your/data/parallel/test/all.csv \
    /path/to/your/data/parallel/test/test_processed.tsv


# ----------------------------------------------------------
# 3. Prepare Training Environment
# ----------------------------------------------------------
# This comand line sets up environment variables for the model, processor, etc.
multimodalhugs-setup --modality "signwriting2text" --config_path $CONFIG_PATH

# ----------------------------------------------------------
# 4. Train the Model
# ----------------------------------------------------------
multimodalhugs-train \
    --task "translation" \
    --config-path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH