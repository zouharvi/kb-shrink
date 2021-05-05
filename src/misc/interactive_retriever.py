#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.embedding import BertWrap
import argparse
from misc.utils import load_dataset, read_keys_pickle
import numpy as np
import scann

parser = argparse.ArgumentParser(description='Interactive retrieval prompt')
parser.add_argument('--dataset', default="data/eli5-dev.jsonl")
parser.add_argument('--keys-in', default="data/eli5-dev.embd")
parser.add_argument('--embd-model', default="bert-base-cased")
parser.add_argument('--type-out', default="pooler")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

print("Loading data")
dataset = load_dataset(args.dataset, keep="inputs")
data = read_keys_pickle(args.keys_in)
data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]
bert = BertWrap(args.embd_model)
print("Finished")

print("Building search tree")
searcher = scann.scann_ops_pybind.builder(data, 20, "dot_product").tree(
    num_leaves=1000,  # TODO: this is number of clusters?
    num_leaves_to_search=150,
    training_sample_size=250000
).score_ah(
    2,
    anisotropic_quantization_threshold=0.2
).reorder(100).build()
print("Finished")

while True:
    query = input("Ask:   ")
    query_embd = bert.sentence_embd(query, args.type_out)
    neighbours, distances = searcher.search(query_embd)
    print(f"{distances[0]:5.2f}:", dataset[neighbours[0]]["input"])