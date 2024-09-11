#!/bin/bash

DATASETS=(
    'nq-dev'
    'scifact-dev'
    'fiqa-dev'
    'hotpotqa-dev'
    'dbpedia-entity-dev'
    'quora-dev'
    'fever-dev'
    'nfcorpus-dev'
)

for file_path in llama3.1-instruct llama3.1; do # reproduced-v2 joint-full 
    for dataset in "${DATASETS[@]}"; do
        echo "Symlinking ${dataset}..."
        short_dataset=$(echo $dataset | sed 's/-dev//')
        echo "bash ./scripts/symlink_msmarco.sh $file_path/$short_dataset $file_path/$dataset"
        bash ./scripts/symlink_msmarco.sh $file_path/$short_dataset $file_path/$dataset
    done
done