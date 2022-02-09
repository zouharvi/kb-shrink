#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, save_json
import argparse
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/text-pruned.pkl")
parser.add_argument('--logfile', default="computed/tmp.json")
args = parser.parse_args()

data = read_pickle(args.data)

print(
    f'Average number of chars per span: {np.average([len(x) for x in data["docs"]]):.2f}'
)

counts_w = Counter([len(x.split(" ")) for x in data["docs"]])
counts_c = Counter([len(x) for x in data["docs"]])
save_json(
    args.logfile, 
    {"words": counts_w, "chars": counts_c}
)