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
OUTPUT_PATH="/generate" # Indicate the path where you would like to store the training actors and model chekpoints
CKPT_PATH="/output/checkpoint-<ckpt_number>" # Choose the preferred checkpoint to evaluate

# ----------------------------------------------------------
# Evaluate using multimodalhugs-generate
# ----------------------------------------------------------

multimodalhugs-generate \
    --task "translation" \
    --config_path $CONFIG_PATH \
    --model_name_or_path $CKPT_PATH \
    --metric_name "sacrebleu" \
    --output_dir $OUTPUT_PATH \
    --do_predict true \
    --generation_max_length 250 \
    --visualize_prediction_prob 0 \
    --report_to none

# ----------------------------------------------------------
# Use multimodalhugs-generate output to evaluate on chrf
# ----------------------------------------------------------

python ../../../scripts/compute_metrics_from_predictions_labels.py \
    --metric_name "chrf" \
    --predictions_labels "${OUTPUT_PATH}/predictions_labels.txt"
