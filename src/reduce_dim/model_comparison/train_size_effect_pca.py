#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse
import timeit
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
    )
    train_time = timeit.timeit(lambda: model.fit(data["docs"][:threshold]), number=1)

    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    
    dataReduced = center_data(dataReduced)
    dataReduced = norm_data(dataReduced)

    # we don't need L2 because the result on normalized data is identical to IP
    val_ip_pca = rprec_ip(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    print(f"threshold: {threshold}, ip: {val_ip_pca:.4f}, train_time: {train_time:.2f}")
    logdata.append({"threshold": threshold, "val_ip": val_ip_pca, "type": "pca", "train_time": train_time})

    # override dump
    with open(args.logfile, "w") as f:
        f.write(str(logdata))