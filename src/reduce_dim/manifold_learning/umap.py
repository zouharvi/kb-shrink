#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.load_utils import read_pickle
from misc.retrieval_utils import rprec_l2, rprec_ip
import numpy as np
import argparse
from umap import UMAP

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c-5000.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

print("Preparing model")
model = UMAP(
    n_components=128, random_state=args.seed,
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
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (UMAP)")
