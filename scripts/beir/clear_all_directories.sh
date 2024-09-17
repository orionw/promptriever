#!/bin/bash

for dataset in nq; do 
    bash scripts/beir/clear_directory.sh reproduced-v2 $dataset-dev
done