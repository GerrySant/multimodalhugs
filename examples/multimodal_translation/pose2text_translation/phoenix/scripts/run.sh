#! /bin/bash

# these are variables you most definitely need to modify:

base="/shares/sigma.ebling.cl.uzh/mathmu/multimodalhugs-examples"

scripts="/shares/sigma.ebling.cl.uzh/mathmu/multimodalhugs/examples/multimodal_translation/pose2text_translation/phoenix/scripts"

dry_run="false"

##########################################################

# preprocess data

. $scripts/phoenix_dataset_preprocessing.sh \
    $base $dry_run $scripts

# HF train

. $scripts/train_phoenix.sh \
    $base $dry_run $scripts


# HF translate + evaluate

.$scripts/translate_phoenix.sh \
    $base
