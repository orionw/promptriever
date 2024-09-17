#!/bin/bash

# example usage:
#   bash scripts/beir/run_all_prompts.sh orionweller/repllama-reproduced-v2 reproduced-v2
#   bash scripts/beir/run_all_prompts.sh orionweller/repllama-instruct-hard-positives-v2-joint joint-full

# export CUDA_VISIBLE_DEVICES="0,1,2,3"

retriever_name=$1
nickname=$2
base_model=$3

# if base model is empty, use meta-llama/Llama-2-7b-hf 
if [ -z "$base_model" ]; then
    base_model="meta-llama/Llama-2-7b-hf"
fi

echo "Retriever name: $retriever_name"
echo "Nickname: $nickname"
echo "Base model: $base_model"

mkdir -p $nickname

datasets=(
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'arguana'
    'hotpotqa'
    'fever'
    'climate-fever'
    'dbpedia-entity'
    'nfcorpus-dev'
    # 'nq-dev'
    'scifact-dev'
    'fiqa-dev'
    'hotpotqa-dev'
    'dbpedia-entity-dev'
    'quora-dev'
    'fever-dev'
    'msmarco'
    "cqadupstack-android"
    "cqadupstack-english"
    "cqadupstack-gaming"
    "cqadupstack-gis"
    "cqadupstack-wordpress"
    "cqadupstack-physics"
    "cqadupstack-programmers"
    "cqadupstack-stats"
    "cqadupstack-tex"
    "cqadupstack-unix"
    "cqadupstack-webmasters"
    "cqadupstack-wordpress"
)


# Read in each line of the generic_prompts.csv file where each line is a prompt
# Run it on each dataset, hashing the prompt and passing that as the fourth argument
gpu_num=0
gpu_max=7
while IFS= read -r prompt
do
    for dataset in "${datasets[@]}"; do
        echo "Running prompt on dataset: $dataset"
        echo "Prompt: '$prompt'"
        # if the gpu_num is the max, don't run it in the background, otherwise run in the background
        if [ $gpu_num -eq $gpu_max ]; then
            bash scripts/beir/encode_beir_queries.sh $base_model "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt"
            # echo "Sleeping for 120 seconds..."
            # sleep 10
            # echo "Done sleeping."
        else
            bash scripts/beir/encode_beir_queries.sh $base_model "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt" &
        fi
        # update the GPU num looping if it hits the max
        gpu_num=$((gpu_num+1))
        if [ $gpu_num -gt $gpu_max ]; then
            gpu_num=0
        fi
    done
done < generic_prompts.csv


# also run one without a prompt for each dataset
for dataset in "${datasets[@]}"; do
    echo "Running without prompt on dataset: $dataset"
    bash scripts/beir/encode_beir_queries.sh $base_model "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num"
    # update the GPU num looping if it hits the max
    gpu_num=$((gpu_num+1))
    if [ $gpu_num -gt $gpu_max ]; then
        gpu_num=0
    fi
done
