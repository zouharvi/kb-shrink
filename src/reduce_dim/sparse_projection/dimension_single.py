#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.load_utils import read_pickle
from misc.retrieval_utils import rprec_ip, rprec_l2
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

print(f"{'Method':<12} {'(IP)':<8} {'(L2)':<8}")

def summary_performance_custom(name, acc_val_ip, acc_val_l2):
    print(f"{name:<12} {acc_val_ip:<8.5f} {acc_val_l2:<8.5f}")


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
    val_ip = rprec_ip(
        dataReduced["queries"].copy(), dataReduced["docs"].copy(), data["relevancy"], report=False, fast=True
    )
    val_l2 = rprec_l2(
        dataReduced["queries"].copy(), dataReduced["docs"].copy(), data["relevancy"], report=False, fast=True
    )

    data_log.append({"dim": dim, "val_ip": val_ip, "val_l2": val_l2})
    
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

    summary_performance_custom(
        f"Dim {dim}", val_ip, val_l2
    )

random_projection_performance(False)
for dim in range(768):
    random_projection_performance(dim)