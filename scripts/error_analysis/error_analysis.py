import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytrec_eval
import seaborn as sns
from tqdm import tqdm
import ir_datasets
from scipy import stats as scipy_stats

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

def load_run(file_path: str) -> Dict[str, Dict[str, float]]:
    run = defaultdict(dict)
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            run[query_id][doc_id] = float(score)
    return run

def load_dataset(dataset_name: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    try:
        dataset = ir_datasets.load(f'beir/{dataset_name}/test')
    except Exception:
        dataset = ir_datasets.load(f'beir/{dataset_name}')

    qrels = defaultdict(dict)
    queries = {}

    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    return qrels, queries

def evaluate_runs(run1: Dict[str, Dict[str, float]], run2: Dict[str, Dict[str, float]], 
                  qrels: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10'})
    
    results1 = evaluator.evaluate(run1)
    results2 = evaluator.evaluate(run2)
    
    return results1, results2

def compare_runs(results1: Dict[str, Dict[str, float]], results2: Dict[str, Dict[str, float]]) -> List[Dict]:
    comparison = []
    for query_id in results1.keys():
        score1 = results1[query_id]['ndcg_cut_10']
        score2 = results2[query_id]['ndcg_cut_10']
        diff = score2 - score1
        if diff == 0:
            label = 'tie'
            continue
        else:
            label = 'run2' if diff > 0 else 'run1'
        comparison.append({
            'query_id': query_id,
            'diff': diff,
            'label': label
        })

    # sort comparison by label
    comparison.sort(key=lambda x: x['label'])
    return comparison

def save_jsonl(data: List[Dict], output_file: str):
    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def plot_statistics(stats: Dict[str, Dict[str, List[float]]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for stat_name, stat_data in stats.items():
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[stat_data['run1'], stat_data['run2']])
        plt.title(f'{stat_name.replace("_", " ").title()} Distribution')
        plt.xticks([0, 1], ['Run 1', 'Run 2'])
        plt.ylabel(stat_name.replace('_', ' ').title())
        plt.savefig(os.path.join(output_dir, f'{stat_name}_boxplot.png'))
        plt.close()

        # Add histogram for this statistic
        plt.figure(figsize=(10, 6))
        sns.histplot(data=stat_data, kde=True)
        plt.title(f'{stat_name.replace("_", " ").title()} Distribution')
        plt.xlabel(stat_name.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{stat_name}_histogram.png'))
        plt.close()



def compute_statistics(queries: Dict[str, str], comparison: List[Dict], results1: Dict[str, Dict[str, float]], results2: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, List[float]]]:
    stats = defaultdict(lambda: defaultdict(list))
    stop_words = set(stopwords.words('english'))
    
    for item in comparison:
        query = queries[item['query_id']]
        label = item['label']
        
        # Existing statistics
        stats['length'][label].append(len(query))
        stats['question_marks'][label].append(query.count('?'))
        stats['exclamation_marks'][label].append(query.count('!'))
        stats['commas'][label].append(query.count(','))
        stats['word_count'][label].append(len(query.split()))
        
        # New statistics
        words = word_tokenize(query.lower())
        words_no_stop = [w for w in words if w not in stop_words]
        
        # Average word length
        stats['avg_word_length'][label].append(np.mean([len(w) for w in words]))
        
        # Unique words
        stats['unique_words'][label].append(len(set(words)))
        
        # Part-of-speech distribution
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter(tag for word, tag in pos_tags)
        stats['noun_count'][label].append(pos_counts.get('NN', 0) + pos_counts.get('NNS', 0))
        stats['verb_count'][label].append(pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0))
        stats['adj_count'][label].append(pos_counts.get('JJ', 0))
        
        # Performance-based metrics
        stats['abs_diff'][label].append(abs(item['diff']))
        stats['rel_improvement'][label].append((item['diff'] / results1[item['query_id']]['ndcg_cut_10']) * 100 if results1[item['query_id']]['ndcg_cut_10'] != 0 else 0)
    
    # Query difficulty estimation (across all queries)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([queries[item['query_id']] for item in comparison])
    query_idf = np.array(tfidf_matrix.sum(axis=0)).flatten()
    
    for i, item in enumerate(comparison):
        label = item['label']
        stats['avg_idf'][label].append(np.mean(query_idf[i]))
    
    return stats

def main(args):
    # make the output directories and plots
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    # Load data
    run1 = load_run(args.run1)
    run2 = load_run(args.run2)
    qrels, queries = load_dataset(args.dataset_name)
    
    # Evaluate runs
    results1, results2 = evaluate_runs(run1, run2, qrels)
    
    # Compare runs
    comparison = compare_runs(results1, results2)
    
    # Add query text to comparison
    for item in comparison:
        item['query'] = queries[item['query_id']]
    
    # Save comparison to JSONL file
    save_jsonl(comparison, os.path.join(args.output_dir, 'comparison.jsonl'))
    
    # Compute statistics
    stats = compute_statistics(queries, comparison, results1, results2)
    
    # Plot statistics
    plot_statistics(stats, os.path.join(args.output_dir, 'plots'))
    
    # Print summary statistics
    print("Summary Statistics:")
    for stat_name, stat_data in stats.items():
        print(f"\n{stat_name.replace('_', ' ').title()}:")
        for run, values in stat_data.items():
            print(f"  {run}: Mean = {np.mean(values):.2f}, Median = {np.median(values):.2f}, "
                  f"Min = {np.min(values):.2f}, Max = {np.max(values):.2f}")
    
    # Perform statistical significance tests
    for stat_name, stat_data in stats.items():
        if 'run1' in stat_data and 'run2' in stat_data:
            t_stat, p_value = scipy_stats.ttest_ind(stat_data['run1'], stat_data['run2'])
            print(f"\n{stat_name.replace('_', ' ').title()} - T-test results:")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two IR model run files")
    parser.add_argument("run1", help="Path to the first run file")
    parser.add_argument("run2", help="Path to the second run file")
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("output_dir", help="Path to the output folder")
    args = parser.parse_args()
    
    main(args)
    # example usage: python scripts/error_analysis.py joint-full/scifact/rank.scifact.trec reproduced-v2/scifact/rank.scifact.trec scifact error_analysis/