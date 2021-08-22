#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data
from misc.retrieval_utils import rprec_ip, rprec_l2
import numpy as np
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--step', type=int, default=10000)
parser.add_argument('--n', default=5000, help="Queries base")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

data["queries"] = data["queries"][:args.n]
data["relevancy"] = data["relevancy"][:args.n]
max_base_doc = max([x for rel_l in data["relevancy"] for x in rel_l])
print(f"Making {(len(data['docs'])-max_base_doc)//args.step} steps from {max_base_doc} (base) to {len(data['docs'])} (total doc count)")

logdata = []

for threshold in range(max_base_doc, len(data['docs'])+args.step-1, args.step):
    docsCropped = data["docs"][:threshold]

    val_ip = rprec_ip(
        data["queries"], docsCropped, data["relevancy"], fast=True
    )
    val_l2 = rprec_l2(
        data["queries"], docsCropped, data["relevancy"], fast=True
    )
    print(f"threshold: {threshold}, ip: {val_ip:.4f}, l2: {val_l2:.4f}")
    logdata.append({"threshold": threshold, "val_ip": val_ip, "val_l2": val_l2, "type": "uncompressed"})
    
    model = PCA(
        n_components=64,
        random_state=args.seed
    ).fit(np.concatenate((data["queries"], docsCropped)))

    # print(f"PCA avg norm before: {np.average(np.linalg.norm(model.components_, axis=1)):.50f}")
    # model.components_ /= np.linalg.norm(model.components_.T, axis=1)[:, np.newaxis].T
    # print(f"PCA avg norm after:  {np.average(np.linalg.norm(model.components_, axis=1)):.50f}")

    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(docsCropped)
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
