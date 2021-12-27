#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data
from misc.retrieval_utils import rprec_a_ip, rprec_a_l2
import argparse
from sklearn import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

print("Preparing model")
model = cluster.FeatureAgglomeration(
    n_clusters=128
)

print("Fitting model")
model.fit(data["docs"])
dataNew = {
    "docs": model.transform(data["docs"]),
    "queries": model.transform(data["queries"]),
}

val_ip_pca = rprec_a_ip(
    dataNew["queries"], dataNew["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
val_l2_pca = rprec_a_l2(
    dataNew["queries"], dataNew["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f}")

dataNew = center_data(dataNew)
dataNew = norm_data(dataNew)

val_ip_pca = rprec_a_ip(
    dataNew["queries"], dataNew["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print(f"ip: {val_ip_pca:.4f} (cn)")