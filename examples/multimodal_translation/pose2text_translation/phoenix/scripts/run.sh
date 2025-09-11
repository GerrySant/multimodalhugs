#! /bin/bash

# these are variables you most definitely need to modify:

base="/shares/sigma.ebling.cl.uzh/mathmu/multimodalhugs-examples"

dry_run="false"

##########################################################

scripts=$(dirname "$0")

# preprocess data

. $scripts/phoenix_dataset_preprocessing.sh \
    $base $dry_run $scripts

# HF train

. $scripts/train_phoenix.sh \
    $base $dry_run $scripts


# HF translate + evaluate

. $scripts/translate_phoenix.sh \
    $base
