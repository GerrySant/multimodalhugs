#! /bin/bash

# calling script needs to set:
# $base

base=$1

configs=$base/configs
configs_sub=$configs/phoenix

models=$base/models
models_sub=$models/phoenix

translations=$base/translations
translations_sub=$translations/phoenix

mkdir -p $translations
mkdir -p $translations_sub

# measure time

SECONDS=0

################################

# check if there are any checkpoints

model_name_or_path=$(ls -d "$models_sub"/train/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)

if [ -z "$model_name_or_path" ]; then
  echo "No checkpoints found in $models_sub"
  exit 1
fi

multimodalhugs-generate \
    --task "translation" \
    --config_path $configs_sub/config_phoenix.yaml \
    --metric_name "sacrebleu" \
    --output_dir $translations_sub \
    --setup_path $models_sub/setup \
    --model_name_or_path $models_sub/train/checkpoint-best \
    --num_beams 5
