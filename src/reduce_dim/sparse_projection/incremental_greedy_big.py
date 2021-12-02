#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_ip, rprec_a_l2
import numpy as np
import argparse

raise Exception("This script has not yet been tried")

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--logfile-single', default="computed/dimension_drop_single.log")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--dims', default="custom")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

with open(args.logfile_single, "r") as f:
    DATA_SINGLE = eval(f.read())
DATA_BASE = [x for x in DATA_SINGLE if x["dim"] == False][0]
IMPR_L2 = [x["dim"] for x in sorted(DATA_SINGLE, key=lambda x: x["val_l2"], reverse=True)]

print("l2_impr count", len([x for x in DATA_SINGLE if x["val_l2"] >= DATA_BASE["val_l2"]]))


class DropRandomProjection():
    def transform(self, data, dim, impr_array):
        return np.delete(np.array(data), impr_array[:dim], axis=1)

data_log = []

def random_projection_performance(data, dim):
    model = DropRandomProjection()
    
    for i, _ in enumerate(data["queries"]):
        data["queries"][i] = model.transform([data["queries"]], dim, IMPR_L2)[0]
    for i, _ in enumerate(data["docs"]):
        data["docs"][i] = model.transform([data["docs"]], dim, IMPR_L2)[0]

    if args.post_cn:
        data = center_data(data)
        data = norm_data(data)

    # copy to make it C-continuous
    val_l2 = rprec_a_l2(
        data["queries"],
        data["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        report=False,
        fast=True,
    )
    if not args.post_cn:
        val_ip = rprec_a_ip(
            data["queries"],
            data["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            report=False,
            fast=True,
        )
    else:
        val_ip = val_l2

    data_log.append({"del_dim": dim, "val_ip": val_ip, "val_l2": val_l2})
    
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

    print(f"Delete {dim} dims: {val_l2:<8.5f}")

DIMS = process_dims()

for dim in DIMS:
    dim = 768 - int(dim)
    data = read_pickle(args.data)
    # take only dev queries
    data = sub_data(data, train=False, in_place=True)
    random_projection_performance(data, dim)
