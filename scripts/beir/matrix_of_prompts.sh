#!/bin/bash

# infinite loop 
# while true; do
    # bash scripts/beir/run_all_prompts.sh orionweller/repllama-instruct-hard-positives-v2-joint joint-full
    # bash scripts/beir/run_all_prompts.sh orionweller/repllama-reproduced-v2 reproduced-v2
    # bash scripts/beir/run_all_prompts.sh orionweller/repllama-instruct-llama3.1 llama3.1 meta-llama/Meta-Llama-3.1-8B
    bash scripts/beir/run_all_prompts.sh orionweller/repllama-instruct-llama3.1-instruct llama3.1-instruct meta-llama/Meta-Llama-3.1-8B-Instruct
    # redo climate-fever llama3.1
    # bash scripts/beir/bm25/prompt_all_bm25.sh 

    # now do the search
    # export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    # bash scripts/beir/search_all_prompts.sh joint-full
    # bash scripts/beir/search_all_prompts.sh reproduced-v2
    # bash scripts/beir/search_all_prompts.sh llama3.1
    bash scripts/beir/search_all_prompts.sh llama3.1-instruct
    # sleep 600
    # bash scripts/beir/bm25/search_all_bm25.sh bm25
# done