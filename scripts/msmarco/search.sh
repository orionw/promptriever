#!/bin/bash

# sleep 14400 && bash scripts/search.sh retriever-llama2
save_path=$1
for dataset in dl19 dl20 dev; do # dev 

    python -m tevatron.retriever.driver.search \
    --query_reps $save_path/${dataset}_queries_emb.pkl \
    --passage_reps "$save_path/"'corpus_emb.*.pkl' \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $save_path/rank.${dataset}.txt

    python -m tevatron.utils.format.convert_result_to_trec --input $save_path/rank.${dataset}.txt \
                                                        --output $save_path/rank.${dataset}.trec \
                                                        --remove_query

    # if dataset is dev use msmarco-passage-dev-subset
    if [ $dataset == "dev" ]; then
        echo "Evaluating ${dataset}..."
        echo "python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset $save_path/rank.${dataset}.trec"
        python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset $save_path/rank.dev.trec > $save_path/rank.dev.eval
    else
        pyserini_dataset="${dataset}-passage"
        echo "Evaluating ${dataset}..."
        echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset $save_path/rank.${dataset}.trec"

        python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset $save_path/rank.${dataset}.trec > $save_path/rank.${dataset}.eval
    fi


done
