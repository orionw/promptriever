#!/bin/bash
# args are (1) name of run (2) dataset, and (3) nodes e.g. "0,1,2,3" (4) port num either 1 or 2 or something

echo "Args are $1 $2 $3 $4"
deepspeed --include localhost:$3 --master_port "6000$4" --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-llama3-$1 \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
  --lora \
  --lora_r 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 500 \
  --dataset_name $2 \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --gradient_accumulation_steps 4 \
  --negatives_first_n 3 
  # --dont_shuffle
  