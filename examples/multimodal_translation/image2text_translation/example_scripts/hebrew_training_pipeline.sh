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
multimodalhugs-setup --modality "image2text" --config_path $CONFIG_PATH

# ----------------------------------------------------------
# 4. Train the Model
# ----------------------------------------------------------
multimodalhugs-train \
    --task "translation" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH
