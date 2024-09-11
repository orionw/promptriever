#!/bin/bash

cd ~/
# if the file doesn't exist, download it
if [ ! -f ./Anaconda3-5.1.0-Linux-x86_64.sh ]; then
    wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
fi
# install manually
# echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
