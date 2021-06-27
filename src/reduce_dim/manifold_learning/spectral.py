#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip
from sklearn.metrics import euclidean_distances
import argparse
from sklearn.manifold import SpectralEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c-1000.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
dataNew = np.array(np.concatenate((data["docs"], data["queries"])))
dataNew -= dataNew.mean()

print("Computing similarities")

print("Preparing model")
model = SpectralEmbedding(n_components=128)

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
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (Spectral)")