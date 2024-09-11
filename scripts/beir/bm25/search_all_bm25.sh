#!/bin/bash

# example: bash scripts/beir/bm25/search_all_bm25.sh bm25

results_path=$1

datasets=(
    'nfcorpus-dev'
    'arguana'
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'hotpotqa'
    'climate-fever'
    'dbpedia-entity'
    'fever'
    # 'msmarco-dl19'
    # 'msmarco-dl20'
    # 'msmarco-dev'
    # 'nq-dev'
    'scifact-dev'
    'fiqa-dev'
    'hotpotqa-dev'
    'dbpedia-entity-dev'
    'quora-dev'
    'fever-dev'
)


evaluate() {
    local dataset_name=$1
    local trec_file=$2
    local output_suffix=$3

    # if the final eval file exists and has a score, skip
    if [[ -f "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval" ]]; then
        # if ndcg_cut_10 is in it or recip in it, skip
        if [[ $(grep -c "ndcg_cut_10" "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval") -gt 0 ]] || [[ $(grep -c "recip_rank" "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval") -gt 0 ]]; then
            echo "Skipping ${dataset_name}${output_suffix} because of existing file ${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
            return
        fi
    fi

    echo "Evaluating ${dataset_name} with ${trec_file}..."

    # if it is not msmarco and not -dev in the name
    if [[ "$dataset_name" != *"msmarco"* ]] && [[ "$dataset_name" != *"-dev"* ]]; then
        python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
        "beir-v1.0.0-${dataset_name}-test" \
        "${trec_file}" \
        > "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
    # else if -dev in the name and is not msmarco
    elif [[ "$dataset_name" == *"-dev"* ]] && [[ "$dataset_name" != *"msmarco"* ]]; then
        # remove the -dev
        new_dataset_name=$(echo $dataset_name | sed 's/-dev//')
        python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
        resources/downloaded/qrels/$new_dataset_name.qrels.sampled \
        "${trec_file}" \
        > "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
    else
        dataset=$(echo $dataset_name | cut -d'-' -f2)
        if [ $dataset == "dev" ]; then
            echo "Evaluating ${dataset}..."
            echo "python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset "${trec_file}""
            python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-passage-dev-subset "${trec_file}" > "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
        else
            pyserini_dataset="${dataset}-passage"
            echo "Evaluating ${dataset}..."
            echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset "${trec_file}""
            python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset "${trec_file}" > "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
        fi
    fi


    echo "Score is saved at ${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
    cat "${results_path}/${dataset_name}/${dataset_name}${output_suffix}.eval"
}

# Process all datasets
for dataset in "${datasets[@]}"; do
    dataset_path="${results_path}/${dataset}"
    
    # Evaluate without prompt
    if [[ -f "${dataset_path}/${dataset}.trec" ]]; then
        evaluate "$dataset" "${dataset_path}/${dataset}.trec" ""
    fi

    # Evaluate with prompts
    for trec_file in "${dataset_path}/${dataset}_"*.trec; do
        if [[ -f "$trec_file" ]]; then
            prompt_hash=$(basename "$trec_file" | sed -n 's/.*_\(.*\)\.trec/\1/p')
            evaluate "$dataset" "$trec_file" "_${prompt_hash}"
        fi
    done
done

# Aggregate results
echo "Aggregating results..."
output_file="${results_path}/bm25_aggregate_results.csv"
echo "Dataset,Prompt,NDCG@10,Recall@100,MRR" > "$output_file"

for dataset in "${datasets[@]}"; do
    dataset_path="${results_path}/${dataset}"
    
    # Process results without prompt
    eval_file="${dataset_path}/${dataset}.eval"
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
    for eval_file in "${dataset_path}/${dataset}_"*.eval; do
        if [[ "$dataset" == "msmarco-dev" ]]; then
            prompt_hash=$(basename "$eval_file" | sed -n 's/.*_\(.*\)\.eval/\1/p')
            mrr=$(awk '/recip_rank / {print $3}' "$eval_file")
            echo "${dataset},${prompt_hash},,,${mrr}" >> "$output_file"
        else
            prompt_hash=$(basename "$eval_file" | sed -n 's/.*_\(.*\)\.eval/\1/p')
            ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
            recall=$(awk '/recall_100 / {print $3}' "$eval_file")
            echo "${dataset},${prompt_hash},${ndcg},${recall}" >> "$output_file"
        fi
    done
done

echo "BM25 aggregate results saved to ${output_file}"