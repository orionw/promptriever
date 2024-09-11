#!/bin/bash

save_path=$1
dataset_name=$2

python -m tevatron.retriever.driver.search \
--query_reps $save_path/${dataset_name}_queries_emb.pkl \
--passage_reps "$save_path/"'corpus_emb.*.pkl' \
--batch_size 1024 \
--depth 1000 \
--save_text \
--save_ranking_to $save_path/rank.${dataset_name}.txt 

python -m tevatron.utils.format.convert_result_to_trec --input $save_path/rank.${dataset_name}.txt \
                                                    --output $save_path/rank.${dataset_name}.trec \
                                                    --remove_query


echo "Evaluating ${dataset_name}..."
echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset_name}-test $save_path/rank.${dataset_name}.trec"
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset_name}-test $save_path/rank.${dataset_name}.trec > $save_path/rank.${dataset_name}.eval
echo "Score is saved at $save_path/rank.${dataset_name}.eval"
cat $save_path/rank.${dataset_name}.eval
