import huggingface_hub
import os
import argparse


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
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    print(f"Uploading {args.folder} to {args.repo}")
    try:
        api.upload_folder(
            folder_path=args.folder,
            repo_id=args.repo,
            repo_type="model",
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
            multi_commits=True,
            multi_commits_verbose=True,
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub")
    parser.add_argument("-f", "--folder", type=str, help="The folder to upload", required=True)
    parser.add_argument("-r", "--repo", type=str, help="The repo to upload to", required=True)
    parser.add_argument("--skip_create", action="store_true", help="Skip creating")
    args = parser.parse_args()
    upload_folder(args)


    # example usage:
    #   python scripts/upload_to_hf.py -f retriever-llama2-4gpu/final -r orionweller/repllama-reproduced-v2