#!/bin/bash
# conda config --set ssl_verify false
# conda create -n tevatron python=3.10 -y
# conda activate tevatron
# git config --global user.email "wellerorion@gmail.com"
# git config --global user.name "Orion Weller"

pip install deepspeed accelerate
pip install transformers datasets peft
pip install faiss-cpu
pip install -r requirements.txt
pip install -e .


git config --global credential.helper store
# huggingface-cli login --token $TOKEN --add-to-git-credential
# conda install -c conda-forge openjdk=11 -y
