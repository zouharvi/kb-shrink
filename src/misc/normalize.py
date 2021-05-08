#!/usr/bin/env python3

import sys; sys.path.append("src")
import argparse
from misc.utils import read_pickle, save_pickle
import numpy as np

parser = argparse.ArgumentParser(description='Normalize embeddings')
parser.add_argument('--data-in', default="data/eli5-dev.embd")
parser.add_argument('--data-out', default="data/eli5-dev-norm.embd")
args = parser.parse_args()

print("Loading")
data = read_pickle(args.data_in)
print("First query element[:4]", data["queries"][0][:4])
print("Normalizing queries")
data["queries"] = data["queries"] / np.linalg.norm(data["queries"], axis=1)[:, np.newaxis]
print("Normalizing docs")
data["docs"] = data["docs"] / np.linalg.norm(data["docs"], axis=1)[:, np.newaxis]
print("First query element[:4]", data["queries"][0][:4])
print("Saving")
save_pickle(args.data_out, data)