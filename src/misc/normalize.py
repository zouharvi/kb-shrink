#!/usr/bin/env python3
import sys; sys.path.append("src")
from misc.load_utils import read_pickle, save_pickle, norm_data, center_data, small_data, zscore_data
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Normalize embeddings')
parser.add_argument('--data-in', default="/data/kilt-hp/dpr-c-5000.embd")
parser.add_argument('--data-out')
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--zscore', action="store_true")
parser.add_argument('--std', action="store_true")
parser.add_argument('--small', type=int)
args,_ = parser.parse_known_args()

assert args.center or args.norm or args.small or args.zscore

print("Loading")
data = read_pickle(args.data_in)
print(
    "First query element[:4]",
    data["queries"][0][:4],
    "Norm:", np.linalg.norm(data["queries"][0])
)

if args.small:
    print(f"Downsizing to {args.small} queries")
    data = small_data(data, args.small)

if args.center:
    print("Centering")
    data = center_data(data)

if args.zscore:
    print("z-scoring")
    data = zscore_data(data)

if args.std:
    print("z-scoring")
    data = zscore_data(data, center=False)

if args.norm:
    print("Normalizing")
    data = norm_data(data)


print(
    "First query element[:4]",
    data["queries"][0][:4],
    "Norm:", np.linalg.norm(data["queries"][0])
)

# pickler does not support serialization for objects over 4GB
data["docs"] = data["docs"].tolist()
data["queries"] = data["queries"].tolist()

print("Saving")
save_pickle(args.data_out, data)
