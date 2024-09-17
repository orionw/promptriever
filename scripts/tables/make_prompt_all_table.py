import csv
from collections import defaultdict
import pandas as pd

# Read the CSV data
data = defaultdict(lambda: defaultdict(float))
datasets = set()
prompts = set()

PRETTY_NAMES = {
    "arguana": "ARG",
    "climate-fever": "CFV",
    "dbpedia-entity": "DBP",
    "fever": "FEV",
    "fiqa": "FQA",
    "hotpotqa": "HQA",
    "nfcorpus": "NFC",
    "nq": "NQ",
    "quora": "QUO",
    "scidocs": "SCD",
    "scifact": "SCF",
    "trec-covid": "COV",
    "webis-touche2020": "TOU"
}

generic_prompts = pd.read_csv("results/generic_prompts.csv_hashes.csv", index_col=None)
generic_prompt_hashes = generic_prompts['prompt_hash'].to_list()
generic_hash_to_text = dict(zip(generic_prompts['prompt_hash'], generic_prompts['prompt']))

with open('results/joint-full_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row['dataset']
        
        # Skip msmarco datasets and those with "-dev" in the name
        if 'msmarco' in dataset or '-dev' in dataset:
            continue
        
        filename = row['filename']
        prompt = filename.split('_')[-1].split('.')[0] if '_' in filename else 'none'
        if prompt not in generic_prompt_hashes:
            continue
        
        # Use ndcg@10 if available, otherwise use mrr
        score = float(row['ndcg@10']) if row['ndcg@10'] else float(row['mrr'])
        
        data[prompt][dataset] = score
        datasets.add(dataset)
        prompts.add(prompt)

# Sort datasets and prompts
sorted_datasets = sorted(datasets)
sorted_prompts = sorted(prompts)
# sorted_prompts = [generic_hash_to_text[hash] for hash in sorted_prompts]

# Generate LaTeX table
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l" + "c" * len(sorted_datasets) + "}\n"
latex_table += "\\hline\nPrompt & " + " & ".join([PRETTY_NAMES[dataset] for dataset in sorted_datasets]) + " \\\\\n\\hline\n"

for prompt in sorted_prompts:
    row = [generic_hash_to_text[prompt]]
    for dataset in sorted_datasets:
        score = data[prompt][dataset]
        row.append(f"{score:.1f}" if score != 0 else "-")
    latex_table += " & ".join(row) + " \\\\\n"

latex_table += "\\hline\n\\end{tabular}\n\\caption{Dataset scores for different prompts}\n\\label{tab:dataset_scores}\n\\end{table}"

# Write the LaTeX table to a file
with open('results/dataset_scores_table.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table has been generated and saved to 'results/dataset_scores_table.tex'")