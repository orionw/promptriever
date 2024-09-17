import os
import csv
import re

def extract_scores(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        scores = {}
        
        patterns = {
            'recall@100': r'recall_100\s+all\s+([\d.]+)',
            'ndcg@10': r'ndcg_cut_10\s+all\s+([\d.]+)',
            'mrr': r'recip_rank\s+all\s+([\d.]+)'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                scores[metric] = float(match.group(1)) * 100  # Convert to percentage
        
        return scores

def extract_hash_from_filename(filename):
    # Split the filename by underscore and take the last part before .eval
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[-1].split('.')[0]
    return 'none'

def process_directory(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.eval'):
                file_path = os.path.join(root, file)
                dataset_name = os.path.basename(file).split("_")[0].replace("rank.", "").replace(".eval", "")
                
                prompt_hash = extract_hash_from_filename(file)
                
                scores = extract_scores(file_path)
                
                result = {
                    'dataset': dataset_name,
                    'prompt_hash': prompt_hash,
                    'filename': file,
                    **scores
                }
                
                results.append(result)
    
    return results

def write_to_csv(results, output_file):
    if not results:
        print(f"No results found for {output_file}. Skipping CSV creation.")
        return

    fieldnames = set(['dataset', 'prompt_hash'])
    for result in results:
        fieldnames.update(result.keys())
    
    fieldnames = sorted(list(fieldnames))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def main():
    directories = ['joint-full', 'bm25', "reproduced-v2", "llama3.1", "llama3.1-instruct", "mistral-v0.1", "mistral-v0.3"]
    results_folder = 'results'
    
    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    for directory in directories:
        results = process_directory(directory)
        output_file = os.path.join(results_folder, f'{os.path.basename(directory)}_results.csv')
        write_to_csv(results, output_file)
        print(f"Results for {directory} written to {output_file}")

if __name__ == "__main__":
    main()