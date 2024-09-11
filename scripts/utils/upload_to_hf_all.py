import huggingface_hub
import os
import argparse
import shutil

def create_final_folder(source_dir):
    final_dir = os.path.join(source_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    files_moved = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isfile(item_path):
            shutil.copy2(item_path, final_dir)
            files_moved.append(item)
    
    return files_moved
        

def upload_folder(args):
    print(f"Creating a new repo {args.repo}")
    api = huggingface_hub.HfApi()
    if not args.skip_create:
        repo_url = api.create_repo(
            args.repo,
            repo_type="model",
            exist_ok=False,
            private=True
        )
    
    # Create the 'final' folder and copy non-folder files into it
    # if there are files in the root of the folder but not folders
    files_not_folders = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]
    if files_not_folders:
        files_moved = create_final_folder(args.folder)
        print(f"Files moved to final folder: {files_moved}")
    else:
        files_moved = None
    
    print(f"Uploading folder structure from {args.folder} to {args.repo}")
    try:
        api.upload_folder(
            folder_path=args.folder,
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=files_moved,  # Ignore all files in the root
            multi_commits=True,
            multi_commits_verbose=True,
        )
    except Exception as e:
        print(e)
        import time
        time.sleep(30)
        print(f"Error Uploading {args.folder} to {args.repo}, trying again")
        api.upload_folder(
            folder_path=args.folder,
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=files_moved,  # Ignore all files in the root
            multi_commits=True,
            multi_commits_verbose=True,

        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a folder structure to Hugging Face Hub")
    parser.add_argument("-f", "--folder", type=str, help="The folder to upload", required=True)
    parser.add_argument("-r", "--repo", type=str, help="The repo to upload to", required=True)
    parser.add_argument("--skip_create", action="store_true", help="Skip creating the repository")
    args = parser.parse_args()
    upload_folder(args)

# Example usage:
# python scripts/upload_to_hf_all.py -f /home/ubuntu/LLaMA-Factory/followir-samaya -r orionweller/followir-samaya
# python scripts/upload_to_hf_all.py -f /home/ubuntu/LLaMA-Factory/followir-samaya-exported -r orionweller/followir-samaya-full


# python scripts/upload_to_hf_all.py -f retriever-llama3-full -r orionweller/followir-joint-llama3.1