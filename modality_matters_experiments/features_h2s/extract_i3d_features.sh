#!/usr/bin/env bash

###############################################################################
# Script: extract_i3d_features.sh
# Description: Run I3D feature extraction on train/val/test splits.
#
# Notes:
# - Paths are configurable via variables below.
# - If running on a SLURM cluster, you can add `#SBATCH` headers at the top.
# - Requires an environment with the necessary Python dependencies and GPU access.
###############################################################################

# [Optional] Uncomment and modify SBATCH headers if running with SLURM:
# #SBATCH --cpus-per-task=12
# #SBATCH --time=48:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --output=/path/to/logs/extract_i3d_features.log

cd "$(dirname "$0")"
set -euo pipefail

### CONFIGURATION ###

# Path to your conda/venv environment
ENV_PATH="/path/to/python/environment/bin/activate"

# Directory containing the feature extraction script and checkpoint
bsl1k_DIR="/path/to/bsl1k"
SCRIPT="${bsl1k_DIR}/i3d_feature_extraction.py" # No need to change
CHECKPOINT="${bsl1k_DIR}/bsl5k.pth.tar" # No need to change


# Dataset base directory
DATASET_BASE="/path/to/How2Sign/How2Sign/sentence_level"

# Video subpath (relative to DATASET_BASE)
VIDEO_SUBPATH="rgb_front/clips_realigned_cropped_and_resized" # No need to change

# Output subpath (relative to DATASET_BASE)
OUTPUT_SUBPATH="rgb_front/clips_realigned_i3d_features" # No need to change

# Common I3D parameters
ENDPOINT="Logits" # No need to change
FPS=23.98 # No need to change
STRIDE=1 # No need to change
NUM_IN_FRAMES=16 # No need to change
BATCH_SIZE=10 # No need to change
NUM_CLASSES=5383 # No need to change

### ENVIRONMENT SETUP ###

# Load modules if needed (example: A100)
# module load a100

# Activate Python environment
source "${ENV_PATH}"

# Create a copy of i3d_feature_extraction.py to the bslk1 repository
cp i3d_feature_extraction.py $SCRIPT

# Change to script directory
cd "${bsl1k_DIR}"

### PROCESS EACH SPLIT ###

for SPLIT in train val test; do
    echo "Extracting features for split: ${SPLIT}"

    VIDEO_PATH="${DATASET_BASE}/${SPLIT}/${VIDEO_SUBPATH}"
    OUTPUT_PATH="${DATASET_BASE}/${SPLIT}/${OUTPUT_SUBPATH}"

    python "${SCRIPT}" \
        --checkpoint_path "${CHECKPOINT}" \
        --endpoint "${ENDPOINT}" \
        --video_path "${VIDEO_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --fps "${FPS}" \
        --stride "${STRIDE}" \
        --num_in_frames "${NUM_IN_FRAMES}" \
        --batch_size "${BATCH_SIZE}" \
        --num_classes "${NUM_CLASSES}"

done

echo "Feature extraction completed for all splits."
