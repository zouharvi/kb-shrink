#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
from model import transform_to_32, transform_to_16, transform_to_8, transform_to_1
import argparse
from sklearn.decomposition import PCA
from itertools import product

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data')
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

COMB_PREC = [
    (1, transform_to_1),
    (8, transform_to_8),
    (16, transform_to_16),
    (32, transform_to_32),
]

COMB_PCA = [
    128, 256, 384, 512, 640, 768
]

def summary_performance(dataReduced):
    if args.post_cn:
        dataReduced = center_data(dataReduced)
        dataReduced = norm_data(dataReduced)

    val_l2 = rprec_a_l2(
        dataReduced["queries"],
        dataReduced["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        fast=True,
    )
    if args.post_cn:
        val_ip = val_l2
    else:
        val_ip = rprec_a_ip(
            dataReduced["queries"],
            dataReduced["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            fast=True,
        )
    return val_ip, val_l2


def performance_mult(data, prec_f, pca_dim):
    if pca_dim is None:
        dataReduced = data
    else:
        model = PCA(n_components=pca_dim)
        model.fit(data["docs"])
        dataReduced = {
            "docs": model.transform(data["docs"]),
            "queries": model.transform(data["queries"]),
        }

    # center data in between
    # actually worsesns the performance slightly
    # dataReduced = center_data(dataReduced)
    # dataReduced = norm_data(dataReduced)

    dataReduced = {
        "queries": prec_f(dataReduced["queries"]),
        "docs": prec_f(dataReduced["docs"])
    }

    return summary_performance(dataReduced)


data = sub_data(data, train=False, in_place=True)

logdata = []
for (prec_bit, prec_f), pca_dim in product(COMB_PREC, COMB_PCA):
    val_ip, val_l2 = performance_mult(data, prec_f, pca_dim)
    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "dim": pca_dim, "bit": prec_bit,
        "compression": 32 / prec_bit * 768 / pca_dim,
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
