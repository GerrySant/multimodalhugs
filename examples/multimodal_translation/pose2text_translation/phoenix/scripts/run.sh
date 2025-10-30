#! /bin/bash

# these are variables you most definitely need to modify:

base="/shares/sigma.ebling.cl.uzh/mathmu/multimodalhugs-examples"

dry_run="false"

##########################################################

# good hyperparameters

learning_rate="1e-5"
warmup_steps=500
label_smoothing_factor="0.1"
gradient_accumulation_steps=3
batch_size=8

##########################################################

scripts=$(dirname "$0")

# preprocess data

. $scripts/phoenix_dataset_preprocessing.sh \
    $base $dry_run $scripts

# HF train

. $scripts/train_phoenix.sh \
    $base $dry_run $scripts \
    $learning_rate $gradient_accumulation_steps $warmup_steps $batch_size $label_smoothing_factor

# HF translate + evaluate

. $scripts/translate_phoenix.sh \
    $base
