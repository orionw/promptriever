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

| Binary | Description |
|:-------|:------------|
| [samaya-ai/promptriever-llama2-7b-v1](https://huggingface.co/samaya-ai/promptriever-llama2-7b-v1) | A Promptriever bi-encoder model based on LLaMA 2 (7B parameters).|
| [samaya-ai/promptriever-llama3.1-8b-instruct-v1](https://huggingface.co/samaya-ai/promptriever-llama3.1-8b-instruct-v1) | A Promptriever bi-encoder model based on LLaMA 3.1 Instruct (8B parameters).|
| [samaya-ai/promptriever-llama3.1-8b-v1](https://huggingface.co/samaya-ai/promptriever-llama3.1-8b-v1) | A Promptriever bi-encoder model based on LLaMA 3.1 (8B parameters).|
| [samaya-ai/promptriever-mistral-v0.1-7b-v1](https://huggingface.co/samaya-ai/promptriever-mistral-v0.1-7b-v1) | A Promptriever bi-encoder model based on Mistral v0.1 (7B parameters). |
| [samaya-ai/RepLLaMA-reproduced](https://huggingface.co/samaya-ai/RepLLaMA-reproduced) | A reproduction of the RepLLaMA model (no instructions). A bi-encoder based on LLaMA 2, trained on the [tevatron/msmarco-passage-aug](https://huggingface.co/datasets/Tevatron/msmarco-passage-aug) dataset. |
| [samaya-ai/msmarco-w-instructions](https://huggingface.co/samaya-ai/msmarco-w-instructions) | A dataset of MS MARCO with added instructions and instruction-negatives, used for training the above models. |


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
