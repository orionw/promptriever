# Promptriever: Retrieval models can be controlled with prompts, just like language models

Official repository for the paper [Promptriever: Retrieval models can be controlled with prompts, just like language models](todo). This repository contains the code and resources for Promptriever, which demonstrates that retrieval models can be controlled with prompts on a per-instance basis, similar to language models. 

Evaluation can also be done by using the MTEB repository, see [here for examples](todo).


## Table of Contents
- [Links](#links)
- [Setup](#setup)
- [Experiments](#experiments)
  - [MSMARCO](#msmarco-experiments)
  - [BEIR](#beir-experiments)
- [Analysis](#analysis)
- [Training](#training)
- [Utilities](#utilities)
- [Citation](#citation)


## Links

| Binary |                                                                 Description                                                                |
|:------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| [promptriever-llama2-v1](https://huggingface.co/jhu-clsp/FollowIR-7B) |  The promptriever dense retrieval model used in the majority of the paper, based on Llama-2  | 
| [msmarco-w-instructions](https://huggingface.co/datasets/jhu-clsp/FollowIR-train) | The dataset used to train promptriever-llama2-v1, from augmenting MSMarco with instruction data and instruction-negatives. |
       

## Setup

To initialize your research environment:

```bash
bash setup/install_conda.sh
bash setup/install_req.sh
python setup/download_dev_sets.py
```

These steps ensure consistent software versions and datasets across all research environments.

## Experiments

### MSMARCO Experiments

Run a complete MSMARCO experiment:

```bash
bash msmarco/encode_corpus.sh <output_path> <model_name>
bash msmarco/encode_queries.sh <output_path> <model_name>
bash msmarco/search.sh <output_path>
```

### BEIR Experiments

Execute comprehensive BEIR experiments:

```bash
bash beir/run_all.sh <model_name> <output_nickname>
bash beir/run_all_prompts.sh <model_name> <output_nickname>
bash beir/search_all_prompts.sh <output_path>
```

The `beir/bm25` subfolder contains scripts for BM25 baseline experiments.

## Analysis

### Visualization

Use scripts in the `plotting` folder to generate insightful visualizations:

- `gather_results.py`: Aggregates results from different runs
- `get_sd_table.py`: Generates standard deviation tables
- `make_prompt_all_table.py`: Creates comprehensive prompt-based result tables
- `make_prompt_table_from_results.py`: Generates detailed tables for prompt effectiveness

### Error Analysis

Conduct in-depth error analysis:

```bash
python error_analysis/error_analysis.py <run1> <run2> <dataset> <output_dir>
```

Additional scripts: `error_analysis_bow.py` and `error_analysis_modeling.py`

## Training

Train or fine-tune retrieval models:

```bash
bash training/train.sh <model_args>
```

Available training scripts:
- `train_instruct_llama3_instruct.sh`
- `train_instruct_llama3.sh`
- `train_instruct_mistral_v1.sh`
- `train_instruct_mistral.sh`
- `train_instruct.sh`

## Utilities

- `utils/symlink_dev.sh` and `utils/symlink_msmarco.sh`: Optimize storage usage
- `utils/upload_to_hf_all.py` and `utils/upload_to_hf.py`: Upload models to Hugging Face Hub
- `utils/validate_all_present.py`: Validate dataset completeness
- `filtering/filter_query_doc_pairs_from_batch_gpt.py`: Implement advanced filtering using GPT model outputs

## Citation

If you found the code, data or model useful, free to cite:

```bibtex
@misc{todo}
}
```