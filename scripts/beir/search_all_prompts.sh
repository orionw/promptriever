#!/bin/bash

# example: bash scripts/beir/search_all_prompts.sh reproduced-v2
# example: bash scripts/beir/search_all_prompts.sh joint-full

save_path=$1

datasets=(
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'arguana'
    'hotpotqa'
    'fever'
    'climate-fever'
    'dbpedia-entity'
    'nfcorpus-dev'
    # 'nq-dev'
    'scifact-dev'
    'fiqa-dev'
    'hotpotqa-dev'
    'dbpedia-entity-dev'
    'quora-dev'
    'fever-dev'
    'msmarco'
    "cqadupstack-android"
    "cqadupstack-english"
    "cqadupstack-gaming"
    "cqadupstack-gis"
    "cqadupstack-wordpress"
    "cqadupstack-physics"
    "cqadupstack-programmers"
    "cqadupstack-stats"
    "cqadupstack-tex"
    "cqadupstack-unix"
    "cqadupstack-webmasters"
    "cqadupstack-wordpress"
)


search_and_evaluate() {
    local dataset_name=$1
    local query_emb_file=$2
    local output_suffix=$3

    # if the final eval file exists and has a score, skip
    if [[ -f "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval" ]]; then
        # if there exists an ndcg_cut_10 in the file or a recip_rank, skip
        if [[ $(grep -c "ndcg_cut_10" "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval") -gt 0 ]] || [[ $(grep -c "recip_rank" "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval") -gt 0 ]]; then
            echo "Skipping ${dataset_name}${output_suffix} because of existing file ${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
            return
        fi
    fi

    echo "Searching and evaluating ${dataset_name} with ${query_emb_file}..."

    python -m tevatron.retriever.driver.search \
    --query_reps "${query_emb_file}" \
    --passage_reps "${save_path}/${dataset_name}/corpus_emb.*.pkl" \
    --batch_size 64 \
    --depth 1000 \
    --save_text \
    --save_ranking_to "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    # if the last command failed, exit
    if [ $? -ne 0 ]; then
        echo "Failed to search ${dataset_name}${output_suffix}"
        exit 1
    fi

    echo "Ranking is saved at ${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    python -m tevatron.utils.format.convert_result_to_trec \
    --input "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt" \
    --output "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
    --remove_query

    # if msmarco is not in the name use beir
    echo "Evaluating ${dataset_name}${output_suffix}..."
    if [[ "$dataset_name" != *"msmarco"* ]] && [[ "$dataset_name" != *"-dev"* ]]; then
        python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
        "beir-v1.0.0-${dataset_name}-test" \
        "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
        > "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    # else if -dev in the name
    elif [[ "$dataset_name" == *"-dev"* ]] && [[ "$dataset_name" != *"msmarco"* ]]; then
        # remove the -dev
        new_dataset_name=$(echo $dataset_name | sed 's/-dev//')
        # echo "NEw dataset name: $new_dataset_name"
        python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
        resources/downloaded/qrels/$new_dataset_name.qrels.sampled \
        "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
        > "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    else
        dataset=$(echo $dataset_name | cut -d'-' -f2)
        if [ $dataset == "dev" ]; then
            echo "Evaluating ${dataset}..."
            echo "python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset $save_path//${dataset_name}/rank.${dataset_name}${output_suffix}.trec"
            python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset $save_path/${dataset_name}/rank.${dataset_name}${output_suffix}.trec > $save_path/${dataset_name}/rank.${dataset_name}${output_suffix}.eval
        else
            pyserini_dataset="${dataset}-passage"
            echo "Evaluating ${dataset}..."
            echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset $save_path/${dataset_name}/rank.${dataset_name}${output_suffix}.trec"
            python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset $save_path/${dataset_name}/rank.${dataset_name}${output_suffix}.trec > $save_path/${dataset_name}/rank.${dataset_name}${output_suffix}.eval
        fi
    fi
    echo "Score is saved at ${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    cat "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    sleep 5
}

# Process all datasets
for dataset in "${datasets[@]}"; do
    dataset_path="${save_path}/${dataset}"
    
    # Search without prompt
    if [[ -f "${dataset_path}/${dataset}_queries_emb.pkl" ]]; then
        search_and_evaluate "$dataset" "${dataset_path}/${dataset}_queries_emb.pkl" ""
    fi

    # Search with generic prompts
    for query_file in "${dataset_path}/${dataset}_queries_emb_"*.pkl; do
        if [[ -f "$query_file" ]]; then
            prompt_hash=$(basename "$query_file" | sed -n 's/.*_emb_\(.*\)\.pkl/\1/p')
            search_and_evaluate "$dataset" "$query_file" "_${prompt_hash}"
        fi
    done
done
#!/bin/bash

# Aggregate results
echo "Aggregating results..."
output_file="${save_path}/aggregate_results.csv"
echo "Dataset,Prompt,NDCG@10,Recall@100,MRR" > "$output_file"

for dataset in "${datasets[@]}"; do
    dataset_path="${save_path}/${dataset}"
    
    # Process results without prompt
    eval_file="${dataset_path}/rank.${dataset}.eval"
    if [[ -f "$eval_file" ]]; then
        if [[ "$dataset" == "msmarco-dev" ]]; then
            mrr=$(awk '/recip_rank / {print $3}' "$eval_file")
            echo "${dataset},no_prompt,,,${mrr}" >> "$output_file"
        else
            ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
            recall=$(awk '/recall_100 / {print $3}' "$eval_file")
            echo "${dataset},no_prompt,${ndcg},${recall}," >> "$output_file"
        fi
    fi
    
    # Process results with prompts
    for eval_file in "${dataset_path}/rank.${dataset}_"*.eval; do
        if [[ -f "$eval_file" ]]; then
            prompt_hash=$(basename "$eval_file" | sed -n 's/.*_\(.*\)\.eval/\1/p')
            if [[ "$dataset" == "msmarco-dev" ]]; then
                mrr=$(awk '/recip_rank / {print $3}' "$eval_file")
                echo "${dataset},${prompt_hash},,,${mrr}" >> "$output_file"
            else
                ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
                recall=$(awk '/recall_100 / {print $3}' "$eval_file")
                echo "${dataset},${prompt_hash},${ndcg},${recall}," >> "$output_file"
            fi
        fi
    done
done

echo "Aggregate results saved to ${output_file}"