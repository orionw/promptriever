import os
import argparse
import logging
import Stemmer
import bm25s.hf
from datasets import load_dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_queries(dataset_name):
    if "msmarco-" in dataset_name:
        logging.info(f"Loading MS MARCO queries for dataset: {dataset_name}")
        dataset = load_dataset(f"tevatron/msmarco-passage", split=dataset_name.split("-")[-1], trust_remote_code=True)
        return {row['query_id']: row['query'] for row in dataset}
    else:
        logging.info(f"Loading queries for dataset: {dataset_name}")
        dataset = load_dataset(f"orionweller/beir", dataset_name, trust_remote_code=True)["test"]
        return {row['query_id']: row['query'] for row in dataset}

def main(args):
    logging.info(f"Starting BM25S search for dataset: {args.dataset_name}")

    # Load the BM25 index from Hugging Face Hub
    if "msmarco-" in args.dataset_name:
        cur_dataset_name = "msmarco"
    elif "-dev" in args.dataset_name:
        cur_dataset_name = args.dataset_name.replace("-dev", "")
    else:
        cur_dataset_name = args.dataset_name
    index_name = f"xhluca/bm25s-{cur_dataset_name}-index"
    logging.info(f"Loading BM25 index from: {index_name}")
    retriever = bm25s.hf.BM25HF.load_from_hub(
        index_name, load_corpus=True, mmap=True
    )
    logging.info("BM25 index loaded successfully")

    # Load queries
    queries = load_queries(args.dataset_name)
    logging.info(f"Loaded {len(queries)} queries")

    # Initialize stemmer
    stemmer = Stemmer.Stemmer("english")
    logging.info("Initialized English stemmer")

    # Prepare output file
    os.makedirs(args.output_dir, exist_ok=True)
    basename = f"{args.dataset_name}_{args.prompt_hash}.trec" if args.prompt_hash else f"{args.dataset_name}.trec"
    output_file = os.path.join(args.output_dir, basename)
    logging.info(f"Results will be saved to: {output_file}")

    with open(output_file, 'w') as f:
        for query_id, query in tqdm(queries.items(), desc="Processing queries"):
            # Append prompt if provided
            if args.prompt.strip != "":
                query += f" {args.prompt}"

            # Tokenize the query
            query_tokenized = bm25s.tokenize([query], stemmer=stemmer)

            # Retrieve the top-k results
            # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
            results, scores = retriever.retrieve(query_tokenized, k=args.top_k)
            # since there is only one query, we can just take the first element
            results = results[0]
            scores = scores[0]

            # Write results in TREC format
            for rank, (doc, score) in enumerate(zip(results, scores)):
                f.write(f"{query_id} Q0 {doc['id']} {rank+1} {score} bm25s\n")

    logging.info(f"Search completed. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25S Search Script")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., webis-touche2020)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt to append to each query")
    parser.add_argument("--prompt_hash", type=str, default="", help="Prompt hash to append to each query")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of top results to retrieve")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    main(args)

    # example usage:
    #   python run_bm25s.py --dataset_name webis-touche2020 --prompt "Retrieve relevant documents for the given query:" --top_k 1000 --output_dir results