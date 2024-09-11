#!/bin/bash

# example usage:
#   bash scripts/beir/bm25/prompt_all_bm25.sh 

nickname=bm25

mkdir -p $nickname

datasets=(
    'arguana'
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'hotpotqa'
    'climate-fever'
    'dbpedia-entity'
    'fever'
    # 'msmarco-dl19'
    # 'msmarco-dl20'
    # 'msmarco-dev'
    'nfcorpus-dev'
    # 'nq-dev'
    'scifact-dev'
    'fiqa-dev'
    'hotpotqa-dev'
    'dbpedia-entity-dev'
    'quora-dev'
    'fever-dev'
)


# Read in each line of the generic_prompts.csv file where each line is a prompt
# Run it on each dataset, hashing the prompt and passing that as the fourth argument
while IFS= read -r prompt
do
    prompt_hash=$(echo -n "$prompt" | md5sum | awk '{print $1}')
    for dataset in "${datasets[@]}"; do
        mkdir -p "$nickname/$dataset"
        if [ -f "$nickname/$dataset/${dataset}_${prompt_hash}.trec" ]; then
            echo "Skipping $dataset because of existing file $nickname/$dataset/$dataaset_$prompt_hash.trec"
            continue
        fi
        echo "Running prompt on dataset: $dataset"
        echo "Prompt: '$prompt'"
        python scripts/beir/bm25/run_bm25s.py --dataset_name "$dataset" --prompt "$prompt" --top_k 1000 --output_dir "$nickname/$dataset" --prompt_hash "$prompt_hash"
    done
done < generic_prompts.csv


# also run one without a prompt for each dataset
for dataset in "${datasets[@]}"; do
echo "Running without prompt on dataset: $dataset"
    if [ -f "$nickname/$dataset/$dataset.trec" ]; then
        echo "Skipping $dataset because of existing file $nickname/$dataset/$dataset.trec"
        continue
    fi
    echo "Running without prompt on dataset: $dataset"
    python scripts/beir/bm25/run_bm25s.py --dataset_name $dataset --top_k 1000 --output_dir "$nickname/$dataset"
done
