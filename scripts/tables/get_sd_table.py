import pandas as pd
import numpy as np
import glob
from collections import defaultdict

SKIP_OLD_HASHES = [
    "0ab0de14665a035b4ce74ea58f0aeb0b", # 
    # "d2b1fa425e0198eb5ba2f9ceaa946389",
    # "bc8581d1f8b9b223247df82aa13707fc",
    # "b09133128f72179896830b2f10a6fa9e",
    "11c51cdccc21293fad66b37e75bbdc94",
    # "eeee229082555a0f22c493370c12651e",
    "476c48e5591c52d8000c65bc88421652" # remove it, match key phrases
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

def process_file(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter out MSMarco and -dev datasets
    df = df[~df['dataset'].str.contains('msmarco|dev', case=False, na=False)]

    # remove old hashes
    df = df[~df['prompt_hash'].isin(SKIP_OLD_HASHES)]
    
    # Group by dataset and calculate standard deviation of ndcg@10
    std_devs = df.groupby('dataset')['ndcg@10'].std()

    # # print the max score;s prompt hash of each dataset, excluding none
    # for dataset in df['dataset'].unique():
    #     # get the scores and prompt hashes for this dataset
    #     cur_set = df[df['dataset'] == dataset]
    #     # find the max score index
    #     max_score_index = cur_set['ndcg@10'].idxmax()
    #     # get the prompt hash for the max score
    #     max_prompt_hash = cur_set.loc[max_score_index, 'prompt_hash']            
    #     print(f"Max prompt hash for {dataset}: {max_prompt_hash}")
    
    # Calculate average standard deviation
    avg_std_dev = std_devs.mean()
    
    return std_devs, avg_std_dev

def create_latex_table(results_bm25, results_reproduced, results_joint):
    latex_table = """
\\begin{table}[h]
\\centering
\\begin{tabular}{lcc}
\\hline
Dataset & BM25 SD & Joint-Full SD \\\\
\\hline
"""
    
    # Combine and sort datasets by average standard deviation in descending order
    all_datasets = set(results_bm25.keys()) | set(results_joint.keys()) | set(results_reproduced.keys())
    all_datasets.remove('Average')
    # sort the datasets by name
    sorted_datasets = sorted(all_datasets)
    
    for dataset in sorted_datasets:
        bm25_sd = results_bm25.get(dataset, 0)
        joint_sd = results_joint.get(dataset, 0)
        reproduced_sd = results_reproduced.get(dataset, 0)
        pretty_name = PRETTY_NAMES.get(dataset, dataset)
        latex_table += f"{pretty_name} & {bm25_sd:.1f} & {reproduced_sd:.1f} & {joint_sd:.1f} \\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += f"Average & {results_bm25['Average']:.1f} & {results_reproduced['Average']:.1f} & {results_joint['Average']:.1f} \\\\\n"
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Standard Deviations of NDCG@10 Scores Across Prompt Hashes}\n"
    latex_table += "\\label{tab:std_devs}\n"
    latex_table += "\\end{table}"
    
    return latex_table

def main(file_paths):
    results = {}
    
    print(file_paths)
    
    for file_path in file_paths:
        if "bm25" in file_path.lower():
            model_name = 'BM25'
        elif "reproduced-v2" in file_path.lower():
            model_name = 'Reproduced-v2'
        elif "joint-full" in file_path.lower():
            model_name = 'Joint-Full'
        else:
            continue
        std_devs, avg_std_dev = process_file(file_path)
        results[model_name] = dict(std_devs)
        results[model_name]['Average'] = avg_std_dev
    
    # Create LaTeX table
    latex_table = create_latex_table(results['BM25'], results["Reproduced-v2"], results['Joint-Full'])
    
    print(latex_table)

# Usage
# read in file paths from results/*.csv
file_paths = list(glob.glob("results/*.csv"))
print(file_paths)
main(file_paths)