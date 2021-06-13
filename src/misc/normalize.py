#!/usr/bin/env python3

import sys; sys.path.append("src")
import argparse
from misc.utils import read_pickle, save_pickle, norm_data, center_data
import numpy as np

parser = argparse.ArgumentParser(description='Normalize embeddings')
parser.add_argument('--data-in', default="data/eli5-dev.embd")
parser.add_argument('--data-out', default="data/eli5-dev-norm.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
args = parser.parse_args()

assert args.center or args.norm

print("Loading")
data = read_pickle(args.data_in)
print("First query element[:4]", data["queries"][0][:4], "Norm:", np.linalg.norm(data["queries"][0]))

if args.center:
    print("Centering")
    data = center_data(data)
    
if args.center:
    print("Normalizing")
    data = norm_data(data)

print("First query element[:4]", data["queries"][0][:4], "Norm:", np.linalg.norm(data["queries"][0]))

print("Saving")
save_pickle(args.data_out, data)