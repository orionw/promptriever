import argparse
import os
import json
import pandas as pd
import tqdm
from datasets import load_dataset


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from peft import PeftModel, PeftConfig


TEMPLATE = """<s> [INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false). Answer using only one word, one of those two choices.

Query: {query}
Document: {text}
Relevant (only output one word, either "true" or "false"): [/INST] """

    

def load_followir(model_name: str = "jhu-clsp/FollowIR-7B"):
    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    token_false_id = tokenizer.get_vocab()["false"]
    token_true_id = tokenizer.get_vocab()["true"]
    tokenizer.model_max_length = 768
    model.config.max_length = 768
    return model, tokenizer, token_true_id, token_false_id




def rank_batch_followir(queries, passages, tokenizer, model, true_token_id, false_token_id):
    assert len(queries) == len(passages)

    prompts = [
        TEMPLATE.format(query=query, text=text) for (query, text) in zip(queries, passages)
    ]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        pad_to_multiple_of=None,
    )

    model = model.cuda()
    with torch.no_grad():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        # calculate the scores by comparing true and false tokens
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, true_token_id]
        false_vector = batch_scores[:, false_token_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores



def get_doc(doc_dict: dict) -> str:
    return (doc_dict.get("title", "") + " " + doc_dict.get("text", "")).strip()


def filter_query_doc_pairs(args):
    print(f"Opening output file {args.output_file}...")
    out_file = open(args.output_file, "w")

   # load batch output file
    print(f"Loading batch output file {args.batch_input}...")
    input_data = []
    with open(args.batch_input, "r") as f:
        for line in f:
            input_data.append(json.loads(line))


    print(f"Loading model...")
    model, tokenizer, true_token_id, false_token_id = load_followir()

    print(f"Scoring...")
    batch_size = args.batch_size
    num_batches = len(input_data) // batch_size + 1
    for idx, batch in tqdm.tqdm(enumerate(range(0, len(input_data), batch_size)), total=num_batches):
        if args.debug and j > 10:
            break
        
        # get the batch
        batch_data = input_data[batch:batch+batch_size]
        batch_queries = [d["query"] + " " + d["instruction"] for d in batch_data]
        passages = [get_doc(d["passage"]) for d in batch_data]
        doc_ids = [d["joint_id"] for d in batch_data]
        scores = rank_batch_followir(batch_queries, passages, tokenizer, model, true_token_id, false_token_id)

        # cache each one by appending to the output_file
        for i, (doc_id, score) in enumerate(zip(doc_ids, scores)):
            out_file.write(f"{doc_id}\t{score:.3f}\n")
            # flush it
            out_file.flush()
    
    out_file.close()

    # read it and print stats
    df = pd.read_csv(args.output_file, sep="\t", names=["doc_id", "score"], index_col=None, header=None)
    print(f"Output file has {len(df)} rows")
    df["group"] = df["doc_id"].apply(lambda x: x.split("_")[-1] if "_" in x else "real")
    df["pred_label"] = df["score"].apply(lambda x: 1 if x > 0.5 else 0)
    # print value counts for each one of the pred_label grouped by group
    print(df.groupby(["group", "pred_label"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--batch_input", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    filter_query_doc_pairs(args)

    # example usage
    #  python scripts/filter_query_doc_pairs_from_batch_gpt -i batch_outputs/batch_instances_Y57xfvrFKYSyxp0SSXIaJXUa.jsonl -o batch_outputs/followir_batch_scores_Y57xfvrFKYSyxp0SSXIaJXUa.tsv
    