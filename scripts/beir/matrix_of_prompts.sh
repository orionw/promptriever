#!/bin/bash

bash scripts/beir/run_all_prompts.sh Samaya-AI/promptriever-llama2-7b-v1 joint-full
bash scripts/beir/run_all_prompts.sh Samaya-AI/RepLLaMA-reproduced reproduced-v2
bash scripts/beir/run_all_prompts.sh Samaya-AI/promptriever-llama3.1-7b-v1 llama3.1 meta-llama/Meta-Llama-3.1-8B
bash scripts/beir/run_all_prompts.sh Samaya-AI/promptriever-llama3.1-7b-instruct-v1 llama3.1-instruct meta-llama/Meta-Llama-3.1-8B-Instruct
bash scripts/beir/run_all_prompts.sh Samaya-AI/promptriever-mistral-v0.3-7b-v1 mistral-v0.3 mistralai/Mistral-7B-v0.3
bash scripts/beir/run_all_prompts.sh Samaya-AI/promptriever-mistral-v0.1-7b-v1 mistral-v0.1 mistralai/Mistral-7B-v0.1
# bash scripts/beir/bm25/prompt_all_bm25.sh 

# now do the search (nneeds 8x40GB VRAM to GPU load MSMarco docs)
bash scripts/beir/search_all_prompts.sh joint-full
bash scripts/beir/search_all_prompts.sh reproduced-v2
bash scripts/beir/search_all_prompts.sh llama3.1
bash scripts/beir/search_all_prompts.sh llama3.1-instruct
bash scripts/beir/search_all_prompts.sh mistral-v0.3
bash scripts/beir/search_all_prompts.sh mistral-v0.1
# bash scripts/beir/bm25/search_all_bm25.sh bm25
