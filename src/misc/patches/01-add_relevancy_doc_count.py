#!/usr/bin/env python3

"""
This patch is necessary so that r-precision may use correct information
regarding the articles.
"""

import sys; sys.path.append("src")
from misc.load_utils import save_pickle, read_pickle
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data-1', default="/data/big-hp/full.pkl")
parser.add_argument('--data-2', default="/data/big-hp/full.embd")
parser.add_argument('--data-out', default="/data/big-hp/full_fixed.embd")
args = parser.parse_args()
data1 = read_pickle(args.data_1)
data2 = read_pickle(args.data_2)

if "relevancy" not in data1:
    raise Exception("First data does not have relevancy entry")
if len(data1["queries"]) != len(data2["queries"]):
    raise Exception("Data lengths (queries) are not matching")
if len(data1["docs"]) != len(data2["docs"]):
    raise Exception("Data lengths (docs) are not matching")
if not all([type(x) in [str, list, np.ndarray, torch.Tensor] for x in data1["queries"]]):
    raise Exception("First data does not contain a simple list/array")
if not all([type(x) is tuple for x in data2["queries"]]) or not all([type(x) is tuple for x in data2["docs"]]):
    raise Exception("Second data does not contain tuples")

print("type1:", type(data1["relevancy"]))
print("type2:", type(data2["relevancy"]))

print("Updating queries")
for query_i, (query2, relevancy_docs, relevancy_articles) in enumerate(data2["queries"]):
    data1["queries"][query_i] = (data1["queries"][query_i], relevancy_docs, relevancy_articles)

print("Updating docs")
for doc_i, (doc2, doc_article) in enumerate(data2["docs"]):
    data1["docs"][doc_i] = (data1["docs"][doc_i], doc_article)

print("Deleting relevancies")
del data1["relevancy"]

save_pickle(args.data_out, data1)