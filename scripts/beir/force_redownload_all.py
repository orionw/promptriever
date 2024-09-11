from datasets import load_dataset
# for dataset in ['NFCorpus', 'FiQA', 'Quora', 'DBPedia-Entity', 'SciFact', 'HotpotQA', "NQ", "FEVER"]:
for dataset in ["NQ"]:
    ds = load_dataset("orionweller/beir", f"{dataset.lower()}-dev", trust_remote_code=True, download_mode="force_redownload")
    print(ds)