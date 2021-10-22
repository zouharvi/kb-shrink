#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle
from misc.retrieval_utils import rprec_a_ip, rprec_a_l2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
data = read_pickle(args.data)
data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

class DropRandomProjection():
    def transform(self, data, dim):
        return np.delete(data, dim, axis=1)

data_log = []

def random_projection_performance(dim):
    if dim == False:
        dataReduced = data
    else:
        model = DropRandomProjection()
        dataReduced = {
            "queries": model.transform(data["queries"], dim),
            "docs": model.transform(data["docs"], dim)
        }

    # copy to make it C-continuous
    val_l2 = rprec_a_l2(
        dataReduced["queries"].copy(),
        dataReduced["docs"].copy(),
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        report=False,
        fast=True,
    )

    data_log.append({"dim": dim, "val_l2": val_l2})
    
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

    print(f"Dim {dim}: {val_l2:<8.5f}")

print(f"{'Method':<12} {'(IP)':<8} {'(L2)':<8}")
random_projection_performance(False)
for dim in range(768):
    random_projection_performance(dim)