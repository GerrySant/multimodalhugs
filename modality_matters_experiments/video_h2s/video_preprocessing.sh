#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Video Cropping and Resizing Script
#
# This script crops and resizes all videos in one or more specified directories.
#
# Features:
# - Optional center-cropping to specified width and height.
# - Resizes videos to desired output dimensions.
# - Optional GPU-accelerated encoding (CUDA, NVENC).
# - Dry-run and progress bar options.
#
# Usage:
#   bash resize_videos.sh
#
# Notes:
# - This script assumes your Python/conda environment is already activated, or 
#   you can adjust the `source` line below to your specific environment path.
# - You may want to run this script in a cluster environment using `sbatch`.
#   If so, wrap this script with an appropriate SLURM submission header.
#
# -----------------------------------------------------------------------------

# Activate Python/FFmpeg environment
source /path/to/your/conda_or_virtualenv/bin/activate

# Cropping (center crop)
USE_CROP=0                                # Set to 1 to enable center cropping
CROP_WIDTH=720                            # Crop width (pixels)
CROP_HEIGHT=720                           # Crop height (pixels)

# Resizing
WIDTH=224                                 # Desired width after resize (pixels)
HEIGHT=224                                # Desired height after resize (pixels)

# Encoding options
USE_GPU=1                                 # Set to 1 to enable GPU-accelerated encoding
GPU_CODEC="h264_nvenc"                    # GPU encoder (e.g., h264_nvenc, hevc_nvenc)

# Debug and UI
DRY_RUN=0                                 # Set to 1 to enable debug mode (prints commands without executing)
PROGRESS_BAR=1                            # Set to 1 to show a progress bar during processing

# ---------------------------
set +e

# IMPORTANT: Modify the paths to point to directory of each slip that contains the realigned clips created using "modality_matters_experiments/create_realigned_clips.py"
for INPUT_DIR in \
"/path/to/train/aligned_video_clips" \
"/path/to/val/aligned_video_clips" \
"/path/to/test/aligned_video_clips"
do
    echo "Processing $INPUT_DIR"

    # Verify input directory exists
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "Error: INPUT_DIR '$INPUT_DIR' is not a valid directory."
        exit 1
    fi

    # Prepare output directory
    BASE_NAME=$(basename "$INPUT_DIR")
    PARENT_DIR=$(dirname "$INPUT_DIR")
    OUTPUT_DIR="${PARENT_DIR}/${BASE_NAME}_cropped_and_resized"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY RUN] Would create directory: '$OUTPUT_DIR'"
    else
        mkdir -p "$OUTPUT_DIR"
    fi

    # Collect video files
    shopt -s nullglob nocaseglob
    files=("$INPUT_DIR"/*.{mp4,mkv,avi,mov})
    total=${#files[@]}
    if [[ $total -eq 0 ]]; then
        echo "No video files found in '$INPUT_DIR'."
        exit 0
    fi

    # Process each video
    for i in "${!files[@]}"; do
        video="${files[i]}"
        filename=$(basename "$video")
        outfile="$OUTPUT_DIR/$filename"
        idx=$((i+1))

        # Show progress bar or dry-run message
        if [[ "$DRY_RUN" -eq 0 && "$PROGRESS_BAR" -eq 1 ]]; then
            bar_length=40
            filled=$(( idx * bar_length / total ))
            empty=$(( bar_length - filled ))
            bar=$(printf "%0.s#" $(seq 1 $filled))
            dash=$(printf "%0.s-" $(seq 1 $empty))
            printf "\r[%s%s] %d/%d %s" "$bar" "$dash" "$idx" "$total" "$filename"
        elif [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[DRY RUN] Processing '$filename'"
        fi

        # Build filter chain
        FILTERS=()
        if [[ "$USE_CROP" -eq 1 ]]; then
            # Center crop: x=(in_w-out_w)/2, y=(in_h-out_h)/2
            FILTERS+=("crop=${CROP_WIDTH}:${CROP_HEIGHT}")
        fi
        FILTERS+=("scale=${WIDTH}:${HEIGHT}")
        FILTER_CHAIN=$(IFS=, ; echo "${FILTERS[*]}")

        # Build ffmpeg command
        if [[ "$USE_GPU" -eq 1 ]]; then
            CMD=(ffmpeg -hwaccel cuda -i "$video" -vf "$FILTER_CHAIN" -c:v "$GPU_CODEC" -c:a copy "$outfile")
        else
            CMD=(ffmpeg -i "$video" -vf "$FILTER_CHAIN" -c:a copy "$outfile")
        fi

        # Execute or print
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[DRY RUN] Command: ${CMD[*]}"
        else
            "${CMD[@]}"
            exit_code=$?
            if [[ $exit_code -ne 0 ]]; then
                echo "Warning: '$filename' failed (exit $exit_code), continuing."
            fi
        fi

    done

    # New line after progress
    if [[ "$DRY_RUN" -eq 0 && "$PROGRESS_BAR" -eq 1 ]]; then
        echo ""
    fi

    # Final summary
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY RUN] Debug simulation complete. No changes made."
    else
        echo "All videos processed and saved to '$OUTPUT_DIR'."
    fi

done
