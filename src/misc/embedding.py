#!/usr/bin/env python3

import json
import pickle
import argparse
from transformers import BertTokenizer, BertModel
from utils import parse_dataset_line

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


tokenizer = BertTokenizer.from_pretrained(args.embd_model)
model = BertModel.from_pretrained(args.embd_model)
model.train(False)

with open(args.dataset, "r") as fread, open(args.embd_out, "wb") as fwrite:
    pickler = pickle.Pickler(fwrite)
    for i, line in enumerate(fread):
        line = parse_dataset_line(line)
        if i % 10 == 0:
            print(i, line["input"])
        encoded_input = tokenizer(line["input"], return_tensors='pt')
        output = model(**encoded_input)[1].detach().numpy()
        pickler.dump(output)