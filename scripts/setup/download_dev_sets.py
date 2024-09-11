import ir_datasets
import json
import os
import argparse
import random
from collections import defaultdict

def process_dataset(dataset_name, output_dir, split='dev'):
    print(f"Processing dataset: {dataset_name}")
    
    # Determine the correct split to use
    if dataset_name.lower() == "nq":
        # use natural-questions/dev instead
        loading_dataset_name = "dpr-w100/natural-questions/dev"
        dataset = ir_datasets.load(loading_dataset_name)
    else:
        try:
            dataset = ir_datasets.load(f"beir/{dataset_name.lower()}/{split}")
        except KeyError:
            print(f"Dev split not found for {dataset_name}, falling back to train split.")
            dataset = ir_datasets.load(f"beir/{dataset_name.lower()}/train")
    
    # Create output directories
    queries_dir = os.path.join(output_dir, 'queries')
    qrels_dir = os.path.join(output_dir, 'qrels')
    os.makedirs(queries_dir, exist_ok=True)
    os.makedirs(qrels_dir, exist_ok=True)
    
    # Save queries
    queries_file = os.path.join(queries_dir, f"{dataset_name.lower()}.dev.jsonl")
    with open(queries_file, 'w') as f:
        for query in dataset.queries_iter():
            json.dump({"query_id": query.query_id, "query": query.text}, f)
            f.write('\n')
    
    # Save qrels
    qrels_file = os.path.join(qrels_dir, f"{dataset_name.lower()}.qrels")
    if dataset_name.lower() == "nq":
        # add "doc" before each docid
        with open(qrels_file, 'w') as f:
            for qrel in dataset.qrels_iter():
                f.write(f"{qrel.query_id} 0 doc{qrel.doc_id} {qrel.relevance}\n")
    else:
        with open(qrels_file, 'w') as f:
            for qrel in dataset.qrels_iter():
                f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
    
    print(f"Saved queries to {queries_file}")
    print(f"Saved qrels to {qrels_file}")
    
    return queries_file

def sample_queries(file, output_file, n):
    all_queries = defaultdict(list)
    query_ids_sampled = set()
    
    dataset_name = os.path.basename(file).split('.')[0]
    with open(file, 'r') as f:
        queries = [json.loads(line) for line in f]
        all_queries[dataset_name] = random.sample(queries, min(n, len(queries)))
    
    with open(output_file, 'w') as f:
        for dataset, queries in all_queries.items():
            for query in queries:
                query['dataset'] = dataset
                json.dump(query, f)
                query_ids_sampled.add(query['query_id'])
                f.write('\n')
    
    print(f"Sampled queries saved to {output_file}")

    # now load and save the qrels for only the sampled queries
    qrels_dir = os.path.dirname(file).replace("/queries", "/qrels")
    with open(os.path.join(qrels_dir, f"{dataset_name}.qrels"), 'r') as f:
        qrels = [line.split() for line in f]
        qrels_sampled = [qrel for qrel in qrels if qrel[0] in query_ids_sampled]
    with open(os.path.join(qrels_dir, f"{dataset_name}.qrels.sampled"), 'w') as f:
        for qrel in qrels_sampled:
            f.write(' '.join(qrel) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Process BEIR datasets and sample queries.")
    parser.add_argument("output_dir", help="Directory to save output files")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of queries to sample from each dataset")
    args = parser.parse_args()
    
    datasets = ['arguana']
    processed_files = [] 
    
    for dataset in datasets:
        processed_files.append(process_dataset(dataset, args.output_dir))

        sample_output = "resources/beir"
        sample_file = os.path.join(sample_output, f'{dataset.lower()}.dev.jsonl')
        sample_queries(processed_files[-1], sample_file, args.sample_size)

if __name__ == "__main__":
    main()
    # python scripts/download_dev_sets.py resources/downloaded 