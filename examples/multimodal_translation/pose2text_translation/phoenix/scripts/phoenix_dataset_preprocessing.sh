#! /bin/bash

# calling script needs to set:
# $base
# $dry_run
# $scripts

base=$1
dry_run=$2
scripts=$3

data=$base/data

poses=$data/poses
preprocessed=$data/preprocessed

mkdir -p $data
mkdir -p $poses $preprocessed

# measure time

SECONDS=0

################################

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
else
    dry_run_arg=""
fi

python $scripts/phoenix_dataset_preprocessing.py \
    --pose-dir $poses \
    --output-dir $preprocessed \
    --tfds-data-dir $data/tensorflow_datasets $dry_run_arg

# sizes
echo "Sizes of preprocessed TSV files:"

wc -l $preprocessed/*

echo "time taken:"
echo "$SECONDS seconds"
