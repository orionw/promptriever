#!/bin/bash

model_name=$1
dataset=$2

rm $model_name/$dataset/rank.${dataset}*
rm $model_name/$dataset/${dataset}*