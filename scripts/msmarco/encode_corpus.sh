#!/bin/bash
path_to_save=$1
model=$2
model_type=$3

# if model type is empty then use llama2
if [ -z "$model_type" ]
then
  model_type="meta-llama/Llama-2-7b-hf"
fi

echo "Path to save: $path_to_save"
echo "Model and Model Type: $model $model_type"

mkdir -p $path_to_save
for s in $(seq -f "%01g" 0 7)
do
# add 4 to the gpu_id
# gpuid=$((s+4))
gpuid=$s
echo $gpuid
CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_type  \
  --lora_name_or_path $model \
  --lora \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 8 \
  --dataset_shard_index ${s} \
  --encode_output_path $path_to_save/corpus_emb.${s}.pkl > logs/msmarco_encode_corpus_$s.log 2>&1 &
done
  
