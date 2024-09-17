#!/bin/bash

# example usage: bash scripts/beir/run_all.sh orionweller/repllama-reproduced-v2 reproduced-v2
# example usage: bash scripts/beir/run_all.sh orionweller/repllama-instruct-hard-positives-v2-joint joint-full
# example usage: bash scripts/beir/run_all.sh orionweller/repllama-instruct-mistral-v0.1 mistral-v1 mistralai/Mistral-7B-v0.1

# bash scripts/beir/run_all.sh orionweller/repllama-instruct-llama3.1-instruct llama3.1-instruct meta-llama/Meta-Llama-3.1-8B-Instruct
# bash scripts/beir/run_all.sh Samaya-AI/promptriever-mistral-v0.1-7b-v1 mistral-v0.1 mistralai/Mistral-7B-v0.1
# bash scripts/beir/run_all.sh Samaya-AI/promptriever-mistral-v0.3-7b-v1 mistral-v0.3 mistralai/Mistral-7B-v0.3


# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

retriever_name=$1
nickname=$2
base_model=$3

echo "retriever_name: $retriever_name"
echo "nickname: $nickname"
echo "base_model: $base_model"

mkdir -p $nickname

datasets=(
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'arguana'
    'hotpotqa'
    'fever'
    'climate-fever'
    'dbpedia-entity'
    'nq'
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
    "msmarco"
)

for dataset in "${datasets[@]}"; do
    # if the dataset already exists (corpus_emb.0.pkl exists), skip it
    if [ -d "$nickname/$dataset" ] && [ -f "$nickname/$dataset/corpus_emb.0.pkl" ]; then
        echo "Skipping $dataset"
        continue
    fi
    echo "Encoding corpus: $dataset"
    bash scripts/beir/encode_beir_corpus.sh $nickname/$dataset $retriever_name $dataset $base_model
done
