#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)

print(f"{'Method':<21} {'Loss-D':<7} {'Loss-Q':<7} {'IPRPR':<0} {'L2RPR':<0}")

def summary_performance(name, dataReduced, dataReconstructed):
    if args.post_cn:
        dataReduced = center_data(dataReduced)
        dataReduced = norm_data(dataReduced)

    val_ip = rprec_ip(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    val_l2 = rprec_l2(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    name = name.replace("float", "f")
    print(f"{name:<21} {val_ip:>5.3f} {val_l2:>5.3f}")
    return val_ip, val_l2

def pca_performance_d(components):
    model = LinearDiscriminantAnalysis(
        n_components=components,
    )
    doc_relevancy = [None]*len(data["docs"])
    for query_i, docs in enumerate(data["relevancy"][:len(data["queries"])]):
        for doc in docs:
            doc_relevancy[doc] = query_i
    model.fit(data["docs"], doc_relevancy)
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    return summary_performance(
        f"LDA",
        dataReduced,
        data
    )

pca_performance_d(args.dim)
