#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/big-hp/dpr-c-pruned.embd_cn")
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()
data = read_pickle(args.data)

matrix_docs = np.cov(np.array(data["docs"]).T, bias=True)
matrix_queries = np.cov(np.array(data["queries"]).T, bias=True)

with open(args.logfile, "w") as f:
    f.write(str({"docs": matrix_docs.tolist(), "queries": matrix_queries.tolist()}))