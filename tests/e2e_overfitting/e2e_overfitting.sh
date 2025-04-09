# #!/usr/bin/env bash

set -e  # Exit on any error

PYTHON_BIN=$(which python)

export CONFIG_PATH='tests/e2e_overfitting/config.yaml'
export OUTPUT_PATH="tests/e2e_overfitting/output_dir"
export GENERATE_PATH="tests/e2e_overfitting/generate_outputs"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=1

rm -rf $OUTPUT_PATH && mkdir -p $OUTPUT_PATH
rm -rf $GENERATE_PATH && mkdir -p $GENERATE_PATH


$PYTHON_BIN multimodalhugs/multimodalhugs_cli/training_setup.py --modality "image2text" --config_path $CONFIG_PATH

$PYTHON_BIN multimodalhugs/multimodalhugs_cli/train.py --task "translation" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH \
    --visualize_prediction_prob 0 \
    --use_cpu \
    --report_to none

CKPT_PATH=$(find $OUTPUT_PATH -maxdepth 1 -type d -name 'checkpoint-*' | \
  sed 's/.*checkpoint-//' | sort -n | tail -1 | \
  xargs -I{} echo "${OUTPUT_PATH}/checkpoint-{}")

$PYTHON_BIN multimodalhugs/multimodalhugs_cli/generate.py \
    --task "translation" \
    --config_path $CONFIG_PATH \
    --model_name_or_path $CKPT_PATH \
    --metric_name "chrf" \
    --output_dir $GENERATE_PATH \
    --do_predict true \
    --use_cpu \
    --generation_max_length 7 \
    --visualize_prediction_prob 0 \
    --report_to none


