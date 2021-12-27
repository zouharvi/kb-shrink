#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import numpy as np
from sklearn.metrics import euclidean_distances
import argparse
from sklearn.manifold import SpectralEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c.embd_cn")
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

val_ip_pca = rprec_a_ip(
    dataNew["queries"], dataNew["docs"], data["relevancy_articles"], data["docs_articles"], data["relevancy"], fast=False
)
val_l2_pca = rprec_a_l2(
    dataNew["queries"], dataNew["docs"], data["relevancy_articles"], data["docs_articles"], data["relevancy"], fast=False
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (Spectral)")

val_ip_pca = rprec_a_ip(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=False
)
val_l2_pca = rprec_a_l2(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=False
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (orig)")
