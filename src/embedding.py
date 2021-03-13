#!/usr/bin/env python3

import json
import pickle
import argparse
from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser(
    description='Compute sentence embeddings of KILT prompts.')
parser.add_argument(
    '--dataset', default="data/eli5-dev.jsonl",
    help='KILT (sub)dataset with JSON lines')
parser.add_argument(
    '--embd-out', default="data/eli5-dev.embd",
    help='Prompt sentence embedding (iterative pickle)')
parser.add_argument(
    '--embd-model', default="bert-base-cased",
    help='Specific BERT model to use (transformers lib)')
args = parser.parse_args()

def load_line(line, keep_answers=False):
    line = json.loads(line)
    if keep_answers:
        return {"input": line["input"], "answer": [x["answer"] for x in line["output"] if "answer" in x]}
    else:
        return {"input": line["input"]}

tokenizer = BertTokenizer.from_pretrained(args.embd_model)
model = BertModel.from_pretrained(args.embd_model)
model.train(False)

with open(args.dataset, "r") as fread, open(args.embd_out, "wb") as fwrite:
    pickler = pickle.Pickler(fwrite)
    for i, line in enumerate(fread):
        line = load_line(line)
        if i % 10 == 0:
            print(i, line["input"])
        encoded_input = tokenizer(line["input"], return_tensors='pt')
        output = model(**encoded_input)[1].detach().tolist()
        pickler.dump(output)