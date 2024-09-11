#!/bin/bash

#  
for dataset in scifact fiqa hotpotqa fever nq dbpedia quora nfcorpus; do 
    bash scripts/beir/clear_directory.sh reproduced-v2 $dataset-dev
done

# also redownload all 
python scripts/beir/force_redownload_all.py