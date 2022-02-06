#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, save_pickle, norm_data, center_data, small_data, zscore_data
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='Normalize embeddings')
parser.add_argument('--data-in', default="/data/hp/dpr-c-pruned.embd_cn")
parser.add_argument('--data-out')
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--std', action="store_true")
parser.add_argument('--small', type=int)
parser.add_argument('--docs-small-random', type=int)
args, _ = parser.parse_known_args()

print("Loading")
data = read_pickle(args.data_in)
print(
    "First query element[:4]",
    data["queries"][0][:4],
    "Norm:", np.linalg.norm(data["queries"][0])
)

if args.small:
    print(f"Downsizing to {args.small} queries")
    print(f"Originally {len(data['docs'])} docs")
    data = small_data(data, args.small)
    print(f"Updated {len(data['docs'])} docs")

if args.docs_small_random:
    print(f"Downsizing to {args.docs_small_random} docs")
    data = {
        "docs": random.sample(data["docs"], args.docs_small_random),
        "queries": []
    }

if args.center:
    print("Centering")
    data = center_data(data)

if args.std:
    print("Dividing by std")
    data = zscore_data(data, center=False)

if args.norm:
    print("Normalizing")
    data = norm_data(data)

if len(data["queries"]) != 0:
    print(
        "First query element[:4]",
        data["queries"][0][:4],
        "Norm:", np.linalg.norm(data["queries"][0])
    )

print("Output shape", data["docs"][0].shape)
# pickler does not support serialization for objects over 4GB
data["docs"] = [x for x in data["docs"]]
data["queries"] = [x for x in data["queries"]]

print("Saving")
save_pickle(args.data_out, data)
