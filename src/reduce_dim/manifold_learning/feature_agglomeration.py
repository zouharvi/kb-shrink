#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip
import argparse
from sklearn import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c-5000.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

print("Preparing model")
model = cluster.FeatureAgglomeration(
    n_clusters=128
)

print("Fitting model")
dataNew = model.fit(data["docs"])
dataNew = {
        "docs": model.transform(data["docs"]),
        "queries": model.transform(data["queries"]),
}

val_ip_pca = rprec_ip(
    data["queries"], data["docs"], data["relevancy"], fast=True
)
val_l2_pca = rprec_l2(
    data["queries"], data["docs"], data["relevancy"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (orig)")

val_ip_pca = rprec_ip(
    dataNew["queries"], dataNew["docs"], data["relevancy"], fast=True
)
val_l2_pca = rprec_l2(
    dataNew["queries"], dataNew["docs"], data["relevancy"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (feature agglomeration)")
