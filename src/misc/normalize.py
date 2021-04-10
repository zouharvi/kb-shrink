#!/usr/bin/env python3

import sys
sys.path.append("src")
import argparse
from misc.utils import read_keys_pickle, save_keys_pickle
import numpy as np

parser = argparse.ArgumentParser(description='Normalize embeddings')
parser.add_argument('--keys-in', default="data/eli5-dev.embd")
parser.add_argument('--keys-out', default="data/eli5-dev-norm.embd")
args = parser.parse_args()

print("Loading")
data = read_keys_pickle(args.keys_in)
print("First element[:4]", data[0][:4])
print("Normalizing")
data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]
print("First element[:4]", data[0][:4])
print("Saving")
save_keys_pickle(data, args.keys_out)