#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, rprec_l2, rprec_ip
from misc.retrieval_utils import read_pickle, rprec_l2, rprec_ip
import numpy as np
from sklearn.metrics import euclidean_distances
import argparse
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c-1000.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
dataNew = np.array(np.concatenate((data["docs"], data["queries"])))
dataNew -= dataNew.mean()

print("Computing similarities")
# similarities = euclidean_distances(dataNew)

print("Preparing model")
model = MDS(
    n_components=128, max_iter=10, eps=1e-9, random_state=args.seed,
    # dissimilarity="precomputed",
    n_jobs=6
)

print("Fitting model")
# dataNew = model.fit_transform(similarities)
dataNew = model.fit_transform(dataNew)

dataNew = {
    "docs": dataNew[:len(data["docs"])].copy(),
    "queries": dataNew[len(data["docs"]):].copy(),
}

print(len(dataNew["docs"]))
print(len(dataNew["queries"]))

val_ip_pca = rprec_ip(
    data["queries"], data["docs"], data["relevancy"], fast=False
)
val_l2_pca = rprec_l2(
    data["queries"], data["docs"], data["relevancy"], fast=False
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (orig)")

val_ip_pca = rprec_ip(
    dataNew["queries"], dataNew["docs"], data["relevancy"], fast=False
)
val_l2_pca = rprec_l2(
    dataNew["queries"], dataNew["docs"], data["relevancy"], fast=False
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (MDS)")