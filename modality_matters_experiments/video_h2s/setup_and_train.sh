#!/bin/bash
cd "$(dirname "$0")"

# ----------------------------------------------------------
# Activate Python environment
# ----------------------------------------------------------
ENV_PATH="/path/to/python_env"
source "$ENV_PATH/bin/activate"

# ----------------------------------------------------------
# Define Variables
# ----------------------------------------------------------

# Paths & settings (override via env vars or pass as args)
CONFIG_PATH="config.yaml"
OUTPUT_PATH="/output" # Indicate the path where you would like to store the training actors and model chekpoints

export WANDB_PROJECT
export CUDA_VISIBLE_DEVICES

# ----------------------------------------------------------
# Setup Model, Processor and Dataset
# ----------------------------------------------------------

# Setup experiment
multimodalhugs-setup \
    --modality "video2text" \
    --config_path "$CONFIG_PATH"

# ----------------------------------------------------------
# Train the Model
# ----------------------------------------------------------

multimodalhugs-train --task "translation" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --visualize_prediction_prob 0.3