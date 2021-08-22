#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data
from misc.retrieval_utils import rprec_l2, rprec_ip, DEVICE
import argparse
import timeit
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--logfile', default="computed/pca_time.log")
parser.add_argument('--thresholds-epochs', action="store_true")
parser.add_argument('--step', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

logdata = []
EPOCHS = [
    1000,
    5000,
    10000,
    15000,
    20000,
    40000,
    60000,
    80000,
    100000,
    len(data['docs'])
]

# override if argument passed
if args.thresholds_epochs:
    thresholds = EPOCHS
else:
    thresholds = range(1000, len(data['docs'])+args.step-1, args.step)
    print(f"Making {(len(data['docs'])-1000)//args.step} steps from {1000} (base) to {len(data['docs'])} (total doc count)")


for threshold in EPOCHS:
    threshold = min(threshold, len(data['docs']))

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