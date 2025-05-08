# Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models

Official repository for the paper [Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models](https://arxiv.org/abs/2409.11136). 

This repository contains the code and resources for Promptriever, which demonstrates that retrieval models can be controlled with prompts on a per-instance basis, similar to language models. 

NOTICE: the **MTEB version of Promptriever is broken in v1, please use the v2 branch** which will become the main branch soon. 

## Table of Contents
- [Links](#links)
- [Setup](#setup)
- [Experiments](#experiments)
  - [MSMARCO](#msmarco-experiments)
  - [BEIR](#beir-experiments)
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
bash setup/install_conda.sh # if you don't have conda already
bash setup/install_req.sh
pip install git+https://github.com/orionw/tevatron
```

## Experiments

### MSMARCO Experiments
Run a MSMARCO experiment (DL19, DL20, Dev) with:

```bash
bash msmarco/encode_corpus.sh <output_path> <model_name>
bash msmarco/encode_queries.sh <output_path> <model_name>
bash msmarco/search.sh <output_path>
```

### BEIR Experiments
To reproduce the BEIR experiments you can either use the batch method (running all models):

```bash
bash scripts/beir/matrix_of_corpus.sh
bash scripts/beir/matrix_of_prompts.sh
bash scripts/beir/search_all_prompts.sh <output_path>
```

Or can also run just one model with:

```bash
bash beir/run_all.sh <model_name> <output_nickname>
bash beir/run_all_prompts.sh <model_name> <output_nickname>
bash beir/search_all_prompts.sh <output_path>
```

The `beir/bm25` subfolder contains scripts for BM25 baseline experiments, using [BM25S](https://github.com/xhluca/bm25s).

## Training
To train a Promptriever model, you can use the scripts in `scripts/training/*`:

```bash
bash scripts/training/train.sh <output_name> <dataset_name> <gpu_ids> <port>
```

Available training scripts:
- `train_instruct.sh` (Llama 2)
- `train_instruct_llama3_instruct.sh`
- `train_instruct_llama3.sh`
- `train_instruct_mistral_v1.sh`
- `train_instruct_mistral.sh` (v0.3)

## Utilities
There are a variety of utilities to symlink corpus files (to avoid double storage when doing the dev set optimization), to upload models to Huggingface, and to filter out bad instruction-negatives.

- `utils/symlink_dev.sh` and `utils/symlink_msmarco.sh`: Optimize storage usage
- `utils/upload_to_hf_all.py` and `utils/upload_to_hf.py`: Upload models to Hugging Face Hub
- `utils/validate_all_present.py`: Validate dataset completeness
- `filtering/filter_query_doc_pairs_from_batch_gpt.py`: Implement advanced filtering using GPT model outputs

## Citation

If you found the code, data or model useful, free to cite:

```bibtex
@article{weller2024promptriever,
      title={Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models}, 
      author={Orion Weller and Benjamin Van Durme and Dawn Lawrie and Ashwin Paranjape and Yuhao Zhang and Jack Hessel},
      year={2024},
      eprint={2409.11136},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.11136}, 
}
```
