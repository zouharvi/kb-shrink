#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import rprec_ip, rprec_l2, read_pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--logfile-single', default="computed/dimension_drop_single.log")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
data = read_pickle(args.data)
data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

with open(args.logfile_single, "r") as f:
    DATA_SINGLE = eval(f.read())
DATA_BASE = [x for x in DATA_SINGLE if x["dim"] == False][0]
IMPR_IP = [x["dim"] for x in sorted(DATA_SINGLE, key=lambda x: x["val_ip"], reverse=True)]
IMPR_L2 = [x["dim"] for x in sorted(DATA_SINGLE, key=lambda x: x["val_l2"], reverse=True)]

print("ip_impr count", len([x for x in DATA_SINGLE if x["val_ip"] >= DATA_BASE["val_ip"]]))
print("l2_impr count", len([x for x in DATA_SINGLE if x["val_ip"] >= DATA_BASE["val_ip"]]))

print(f"{'Method':<12} {'(IP)':<8} {'(L2)':<8}")

def summary_performance_custom(name, acc_val_ip, acc_val_l2):
    print(f"{name:<12} {acc_val_ip:<8.5f} {acc_val_l2:<8.5f}")


class DropRandomProjection():
    def transform(self, data, dim, impr_array):
        return np.delete(data, impr_array[:dim], axis=1)

data_log = []

def random_projection_performance(dim, metric):
    if metric == "l2":
        impr_array = IMPR_L2
    elif metric == "ip":
        impr_array = IMPR_IP

    model = DropRandomProjection()
        
    dataReduced = {
        "queries": model.transform(data["queries"], dim, impr_array),
        "docs": model.transform(data["docs"], dim, impr_array)
    }

    # copy to make it C-continuous
    val_ip = rprec_ip(
        dataReduced["queries"].copy(), dataReduced["docs"].copy(), data["relevancy"], report=False, fast=True
    )
    val_l2 = rprec_l2(
        dataReduced["queries"].copy(), dataReduced["docs"].copy(), data["relevancy"], report=False, fast=True
    )

    data_log.append({"del_dim": dim, "val_ip": val_ip, "val_l2": val_l2, "metric": metric})
    
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

    summary_performance_custom(
        f"Delete :{dim}", val_ip, val_l2
    )

for dim in np.linspace(32, 768, num=768//32, endpoint=True):
    dim = int(dim)
    random_projection_performance(dim, "ip")

for dim in np.linspace(32, 768, num=768//32, endpoint=True):
    dim = int(dim)
    random_projection_performance(dim, "l2")
