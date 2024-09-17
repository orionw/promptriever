#!/bin/bash

# infinite loop
while true; do
    # for each folder path in the list, upload to huggingface
    for folder_path in llama3.1 llama3.1-instruct mistral-v0.1 mistral-v0.3 reproduced-v2 joint-full;
    do 
        python scripts/utils/upload_to_hf_folder.py -f $folder_path -r orionweller/promptriever-$folder_path
        # if there was an error sleep for an hour
        if [ $? -ne 0 ]; then
            sleep 1h
        fi
    done
done