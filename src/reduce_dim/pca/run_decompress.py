#!/usr/bin/env python3

import copy
import sys
sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Comparison of original, compressed and decompressed performance')
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--dims', default="custom")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)
data_small = copy.deepcopy(data)

data = sub_data(data, train=False, in_place=True)
data_small = sub_data(data_small, train=True, in_place=True)


def summary_performance(dataIn):
    if args.post_cn:
        dataIn = center_data(dataIn)
        dataIn = norm_data(dataIn)

    val_l2 = rprec_a_l2(
        dataIn["queries"],
        dataIn["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        fast=True,
    )

    if args.post_cn:
        val_ip = val_l2
    else:
        val_ip = rprec_a_ip(
            dataIn["queries"],
            dataIn["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            fast=True,
        )
    
    return val_ip, val_l2


def pca_performance_q(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
        copy=False,
    ).fit(data_small["queries"])

    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }

    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }

    with open("pca_decompress.tmp", "w") as f:
        f.write(str(data["queries"][0].shape))
        f.write(str(dataReduced["queries"][0].shape))
        f.write(str(dataReconstructed["queries"][0].shape))

    return {
        "orig": summary_performance(data),
        "reduced": summary_performance(dataReduced),
        "decompress": summary_performance(dataReconstructed),
    }

DIMS = process_dims(args.dims)

# traverse from large to small
# DIMS.reverse()

logdata = []
for dim in DIMS:
    dim = int(dim)

    vals = pca_performance_q(dim)
    logdata.append(vals)

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
