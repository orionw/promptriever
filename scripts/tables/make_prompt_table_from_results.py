import csv
from collections import defaultdict
import statistics
import os
import random

# old hashes that are unused.
SKIP_OLD_HASHES = [
    "0ab0de14665a035b4ce74ea58f0aeb0b", 
    "11c51cdccc21293fad66b37e75bbdc94",
    "476c48e5591c52d8000c65bc88421652"
]


PRETTY_NAMES = {
    "arguana": "Arguana",
    "climate-fever": "Climate-FEVER",
    "dbpedia-entity": "DBPedia",
    "fever": "FEVER",
    "fiqa": "FiQA",
    "hotpotqa": "HotpotQA",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quora": "Quora",
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "trec-covid": "TREC-COVID",
    "webis-touche2020": "Touche-2020"
}

def read_csv(filename):
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(filename):
        print(f"Warning: {filename} does not exist. Skipping this file.")
        return data
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset'].lower()
            prompt_hash = row['prompt_hash']
            if prompt_hash in SKIP_OLD_HASHES:
                continue
            ndcg = float(row['ndcg@10']) if row['ndcg@10'] else None
            recall = float(row['recall@100']) if 'recall@100' in row and row['recall@100'] else None

            if prompt_hash == 'none':
                data[dataset]['None'] = ndcg
            else:
                data[dataset]['Prompted'][prompt_hash] = (ndcg, recall)
    return data

def format_value(value):
    return f"{value:.1f}" if type(value) == float else "-"

def calculate_average(values):
    real_vals = [v for v in values if v is not None and type(v) == float]
    if not len(real_vals):
        return None
    return statistics.mean(real_vals)

def get_best_dev_prompt(dataset_name, dev_data, is_model: bool = False):
    if not len(dev_data) or not dev_data['Prompted']:
        return None, None
    
    try:
        rounded_scores = {k: (f"{v[0]:.1f}", v[1])  for k, v in dev_data['Prompted'].items()}
    except Exception:
        breakpoint()
    max_ndcg = max(v[0] for v in rounded_scores.values())
    
    # Filter prompts with the highest NDCG
    # prompt can't be None
    best_prompts = {k: v for k, v in rounded_scores.items() if v[0] == max_ndcg and k not in ['none', None]}


    # also print the name of the prompt_hash
    # if is_model:
    #     print(f"Best prompt for {dataset_name} is {best_prompts}")
    
    
    if len(best_prompts) == 1:
        return list(best_prompts.keys())[0], max_ndcg
    
    # If there's a tie, use recall as a tiebreaker
    max_recall = max(v[1] or 0 for v in best_prompts.values())
    best_prompts = {k: v for k, v in best_prompts.items() if v[1] == max_recall}

    if len(best_prompts) == 1:
        return list(best_prompts.keys())[0], max_ndcg
    
    # If there's still a tie, choose the prompt with the lowest hash value
    best_prompt = min(best_prompts.keys())

    return best_prompt, max_ndcg

import csv
from collections import defaultdict
import statistics
import os
import random

def generate_latex_table(bm25_data, repllama_data, modelname_data):
    datasets = list(PRETTY_NAMES.keys())
    latex_rows = []
    
    tuned_datasets = {
        'bm25': set(),
        'repllama': set(),
        'modelname': set()
    }

    for dataset in datasets:
        pretty_name = PRETTY_NAMES[dataset]
        dev_dataset = f"{dataset}-dev"

        bm25_prompt, bm25_prompt_score = get_best_dev_prompt(dev_dataset, bm25_data.get(dev_dataset, {}))
        repllama_prompt, repllama_prompt_score = get_best_dev_prompt(dev_dataset, repllama_data.get(dev_dataset, {}), True)
        modelname_prompt, modelname_prompt_score = get_best_dev_prompt(dev_dataset, modelname_data.get(dev_dataset, {}))

        print(dataset, repllama_prompt)

        bm25_prompt_value = bm25_data[dataset]['Prompted'].get(bm25_prompt, (None, None))[0] if bm25_prompt else None
        repllama_prompt_value = repllama_data[dataset]['Prompted'].get(repllama_prompt, (None, None))[0] if repllama_prompt else None
        modelname_prompt_value = modelname_data[dataset]['Prompted'].get(modelname_prompt, (None, None))[0] if modelname_prompt else None

        if bm25_prompt_value is not None:
            tuned_datasets['bm25'].add(dataset)
        if repllama_prompt_value is not None:
            tuned_datasets['repllama'].add(dataset)
        if modelname_prompt_value is not None:
            tuned_datasets['modelname'].add(dataset)

        row = [
            pretty_name,
            format_value(bm25_data[dataset]['None']),
            format_value(bm25_prompt_value),
            format_value(max(v[0] for v in bm25_data[dataset]['Prompted'].values()) if bm25_data[dataset]['Prompted'] else None),
            format_value(repllama_data[dataset]['None']),
            format_value(repllama_prompt_value),
            format_value(max(v[0] for v in repllama_data[dataset]['Prompted'].values()) if repllama_data[dataset]['Prompted'] else None),
            format_value(modelname_data[dataset]['None']),
            format_value(modelname_prompt_value),
            format_value(max(v[0] for v in modelname_data[dataset]['Prompted'].values()) if modelname_data[dataset]['Prompted'] else None)
        ]
        latex_rows.append(" & ".join(row) + " \\\\")

    averages = [
        "Average",
        format_value(calculate_average([bm25_data[d]['None'] for d in datasets])),
        "-",
        format_value(calculate_average([max(v[0] for v in bm25_data[d]['Prompted'].values()) if bm25_data[d]['Prompted'] else None for d in datasets])),
        format_value(calculate_average([repllama_data[d]['None'] for d in datasets])),
        "-",
        format_value(calculate_average([max(v[0] for v in repllama_data[d]['Prompted'].values()) if repllama_data[d]['Prompted'] else None for d in datasets])),
        format_value(calculate_average([modelname_data[d]['None'] for d in datasets])),
        "-",
        format_value(calculate_average([max(v[0] for v in modelname_data[d]['Prompted'].values()) if modelname_data[d]['Prompted'] else None for d in datasets]))
    ]

    averages_tuned = [
        "Average (Tuned)",
        format_value(calculate_average([bm25_data[d]['None'] for d in tuned_datasets['bm25']])),
        format_value(calculate_average([bm25_data[d]['Prompted'].get(get_best_dev_prompt(f"{d}-dev", bm25_data.get(f"{d}-dev", {}))[0], (None, None))[0] for d in tuned_datasets['bm25']])),
        format_value(calculate_average([max(v[0] for v in bm25_data[d]['Prompted'].values()) if bm25_data[d]['Prompted'] else None for d in tuned_datasets['bm25']])),
        format_value(calculate_average([repllama_data[d]['None'] for d in tuned_datasets['repllama']])),
        format_value(calculate_average([repllama_data[d]['Prompted'].get(get_best_dev_prompt(f"{d}-dev", repllama_data.get(f"{d}-dev", {}))[0], (None, None))[0] for d in tuned_datasets['repllama']])),
        format_value(calculate_average([max(v[0] for v in repllama_data[d]['Prompted'].values()) if repllama_data[d]['Prompted'] else None for d in tuned_datasets['repllama']])),
        format_value(calculate_average([modelname_data[d]['None'] for d in tuned_datasets['modelname']])),
        format_value(calculate_average([modelname_data[d]['Prompted'].get(get_best_dev_prompt(f"{d}-dev", modelname_data.get(f"{d}-dev", {}), True)[0], (None, None))[0] for d in tuned_datasets['modelname']])),
        format_value(calculate_average([max(v[0] for v in modelname_data[d]['Prompted'].values()) if modelname_data[d]['Prompted'] else None for d in tuned_datasets['modelname']]))
    ]

    latex_table = f"""
\\begin{{table*}}[t]
\\centering
\\begin{{tabular}}{{l|ccc|ccc|ccc}}
\\toprule
\\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{3}}{{c|}}{{BM25}} & \\multicolumn{{3}}{{c|}}{{RepLLaMA}} & \\multicolumn{{3}}{{c}}{{\\modelname}} \\\\
\\cmidrule(l){{2-4}} \\cmidrule(l){{5-7}} \\cmidrule(l){{8-10}}
 & None & Prompt & Oracle & None & Prompt & Oracle & None & Prompt & Oracle \\\\
\\midrule
{chr(10).join(latex_rows)}
\\midrule
{" & ".join(averages_tuned)} \\\\
{" & ".join(averages)} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Effectiveness of BM25, RepLLaMA, and \\modelname on BEIR datasets. Results are shown for standard retrieval (None), best prompt from dev set (Prompt), and best overall prompt (Oracle). Missing values are indicated by "-".}}
\\label{{tab:beir-results}}
\\end{{table*}}
"""
    return latex_table

# Main execution
bm25_data = read_csv('results/bm25_results.csv')
repllama_data = read_csv('results/reproduced-v2_results.csv')
modelname_data = read_csv('results/joint-full_results.csv')


latex_table = generate_latex_table(bm25_data, repllama_data, modelname_data)

with open('results/final_table.tex', 'w') as f:
    f.write(latex_table)

print("Final table has been generated and saved as 'results/final_table.tex'")

def remove_prompt_columns_refined(latex_table):
    lines = latex_table.split('\n')
    modified_lines = []

    for line in lines:
        if '\\begin{tabular}' in line:
            # Update the tabular environment
            line = line.replace('{l|ccc|ccc|ccc}', '{l|cc|cc|cc}')
        elif '\\multirow{2}{*}{Dataset}' in line:
            # The header is already correct, keep it as is
            modified_lines.append(line)
        elif 'None & Prompt & Oracle' in line:
            # Remove 'Prompt' from the subheader
            line = line.replace('None & Prompt & Oracle', 'None & Oracle')
        elif '\\cmidrule' in line:
            # The cmidrule specifications are already correct, keep them as is
            modified_lines.append(line)
        elif ' & ' in line and not any(keyword in line for keyword in ['\\midrule', '\\bottomrule', 'Average']):
            # This is a data row, remove 'Prompt' columns
            parts = line.split('&')
            new_parts = [parts[0]] + [parts[i] for i in [1, 3, 4, 6, 7, 9]]
            line = ' & '.join(new_parts)
        elif 'Average' in line:
            # Handle average rows
            parts = line.split('&')
            new_parts = [parts[0]] + [parts[i] for i in [1, 3, 4, 6, 7, 9]]
            line = ' & '.join(new_parts)
        
        modified_lines.append(line)

    return '\n'.join(modified_lines)

# Apply the modification
modified_latex_table = remove_prompt_columns_refined(latex_table)

# Write the modified table to a new file
with open('results/final_table_without_prompt_refined.tex', 'w') as f:
    f.write(modified_latex_table)

print("Modified table without 'Prompt' columns has been generated and saved as 'results/final_table_without_prompt_refined.tex'")
