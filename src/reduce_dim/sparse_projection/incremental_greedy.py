#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_ip, rprec_a_l2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--logfile-single', default="computed/dimension_drop_single.log")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--dims', default="custom")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
data = read_pickle(args.data)

# take only dev queries
data = sub_data(data, train=False, in_place=True)

data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

with open(args.logfile_single, "r") as f:
    DATA_SINGLE = eval(f.read())
DATA_BASE = [x for x in DATA_SINGLE if x["dim"] == False][0]
IMPR_L2 = [x["dim"] for x in sorted(DATA_SINGLE, key=lambda x: x["val_l2"], reverse=True)]

print("l2_impr count", len([x for x in DATA_SINGLE if x["val_l2"] >= DATA_BASE["val_l2"]]))


class DropRandomProjection():
    def transform(self, data, dim, impr_array):
        return np.delete(data, impr_array[:dim], axis=1)

data_log = []

def safe_transform(model, array, dim, impr_array):
    return [model.transform([x], dim, impr_array)[0] for x in array]

def random_projection_performance(dim):
    model = DropRandomProjection()
        
    dataReduced = {
        "queries": safe_transform(model, data["queries"], dim, IMPR_L2),
        "docs": safe_transform(model, data["docs"], dim, IMPR_L2)
    }
    if args.post_cn:
        dataReduced = center_data(dataReduced)
        dataReduced = norm_data(dataReduced)

    # copy to make it C-continuous
    val_l2 = rprec_a_l2(
        dataReduced["queries"],
        dataReduced["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        report=False,
        fast=True,
    )
    if not args.post_cn:
        val_ip = rprec_a_ip(
            dataReduced["queries"],
            dataReduced["docs"],
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

if args.dims == "custom":
    DIMS = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
elif args.dims == "linspace":
    DIMS = np.linspace(32, 768, num=768 // 32, endpoint=True)
else:
    raise Exception(f"Unknown --dims {args.dims} scheme")

# TODO WARN quick hack to continue previous run
DIMS = DIMS[10:]

for dim in DIMS:
    dim = 768 - int(dim)
    random_projection_performance(dim)
