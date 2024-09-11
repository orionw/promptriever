import pandas as pd
import hashlib
import argparse
import sys
import json
from collections import defaultdict

def calculate_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_and_hash_csv(filename, column_name):
    try:
        df = pd.read_csv(filename, header=None, index_col=None)
    except Exception: # load as newlines
        df = pd.read_csv(filename, header=None, index_col=None, sep='\t')
    # set the columns name to prompt if there is only one else dataset,prompt
    df.columns = ['prompt'] if len(df.columns) == 1 else ['dataset', 'prompt']
    # save json of hash to text
    df['prompt_hash'] = df[column_name].apply(calculate_md5)
    df.to_csv(f"results/{filename}_hashes.csv", index=False)
    return set(df[column_name].apply(calculate_md5))

def validate_hashes(data_csv, generic_hashes, domain_hashes):
    df = pd.read_csv(f"results/{data_csv}_results.csv")
    
    dataset_hashes = defaultdict(set)
    for _, row in df.iterrows():
        if pd.notna(row['prompt_hash']) and pd.notna(row['dataset']):
            dataset_hashes[row['dataset']].add(row['prompt_hash'])
    
    print(f"Validation results for {data_csv}:")
    
    for dataset, hashes in dataset_hashes.items():

        
        missing_generic = generic_hashes - hashes
        if missing_generic:
            print(f"\nDataset: {dataset}")
            print(f"Total hashes in dataset: {len(hashes)}")
            print(f"Missing generic hashes: {len(missing_generic)}")
            print(missing_generic)
        
        if dataset in domain_hashes: # not every dataset has domain-specific, e.g. dev sets
            missing_domain = domain_hashes[dataset] - hashes
            if missing_domain:
                print(f"\nDataset: {dataset}")
                print(f"Total hashes in dataset: {len(hashes)}")
                print(f"Missing domain hashes: {len(missing_domain)}")
                print(missing_domain)
        
        # if not missing_generic and (dataset not in domain_hashes or not missing_domain):
        #     print(f"All expected hashes present for dataset {dataset}.")
    
    # Check for datasets in domain_hashes that are not in the results
    missing_datasets = set(domain_hashes.keys()) - set(dataset_hashes.keys())
    if missing_datasets:
        print("\nDatasets missing from results:")
        print(missing_datasets)

def main():
    parser = argparse.ArgumentParser(description="Validate CSV file hashes against generic and domain CSV files.")
    parser.add_argument("file_to_validate", help="The CSV file to validate")
    parser.add_argument("--generic", default="generic_prompts.csv", help="Path to the generic CSV file (default: generic_prompts.csv)")
    parser.add_argument("--domain", default="domain_prompts.csv", help="Path to the domain CSV file (default: domain_prompts.csv)")
    
    args = parser.parse_args()

    try:
        # Load and hash the generic CSV file
        generic_hashes = load_and_hash_csv(args.generic, 'prompt')
        
        # Load and hash the domain CSV file
        domain_df = pd.read_csv(args.domain, header=None, index_col=None)
        domain_df.columns = ["dataset", "prompt"]
        domain_hashes = defaultdict(set)
        hash_map = {}
        for _, row in domain_df.iterrows():
            domain_hashes[row['dataset']].add(calculate_md5(row['prompt']))
            hash_map[calculate_md5(row['prompt'])] = row['prompt']

        # save hash map
        with open("results/hash_map_domain.json", "w") as f:
            json.dump(hash_map, f)

        # Validate the data CSV file
        validate_hashes(args.file_to_validate, generic_hashes, domain_hashes)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {args.file_to_validate} is empty.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

    # python scripts/validate_all_present.py joint-full
    # python scripts/validate_all_present.py bm25