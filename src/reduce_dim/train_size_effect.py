#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--step', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

print(f"Making {(len(data['docs'])-1000)//args.step} steps from {1000} (base) to {len(data['docs'])} (total doc count)")

logdata = []

for threshold in range(1000, len(data['docs'])+args.step-1, args.step):
    model = PCA(
        n_components=128,
        random_state=args.seed
    ).fit(data["docs"][:threshold])

    # print(f"PCA avg norm before: {np.average(np.linalg.norm(model.components_, axis=1)):.50f}")
    # model.components_ /= np.linalg.norm(model.components_.T, axis=1)[:, np.newaxis].T
    # print(f"PCA avg norm after:  {np.average(np.linalg.norm(model.components_, axis=1)):.50f}")

    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    val_ip_pca = rprec_ip(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    val_l2_pca = rprec_l2(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    print(f"threshold: {threshold}, ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (PCA)")
    logdata.append({"threshold": threshold, "val_ip": val_ip_pca, "val_l2": val_l2_pca, "type": "pca"})

    # override dump
    with open(args.logfile, "w") as f:
        f.write(str(logdata))