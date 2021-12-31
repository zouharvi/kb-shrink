#!/usr/bin/env python3

import copy
import sys
sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, sub_data, CenterScaler, NormScaler
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import numpy as np
import argparse
from sklearn.decomposition import PCA
import sklearn.metrics

parser = argparse.ArgumentParser(description='Main PCA performance experiment')
parser.add_argument('--data')
parser.add_argument('--data-train', default=None)
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--dims', default="custom")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

if args.data_train is None:
    print("Because args.data_train is not provided, I'm copying the whole structure")
    data_train = copy.deepcopy(data)
else:
    data_train = read_pickle(args.data_train)

data = sub_data(data, train=False, in_place=True)
data_train = sub_data(data_train, train=True, in_place=True)

if args.center:
    # only keep the dev scaler
    center_model = CenterScaler()
    data = center_model.transform(data)
    data_train = CenterScaler().transform(data_train)
    
if args.norm:
    # only keep the dev scaler
    norm_model = NormScaler()
    data = norm_model.transform(data)
    data_train = NormScaler().transform(data_train)


def summary_performance(dataReduced, dataReconstructed):
    # reconstructed data is not in the original form when scaling
    # note the reverse order
    if args.norm:
        dataReconstructed = norm_model.inverse_transform(dataReconstructed)
    if args.center:
        dataReconstructed = center_model.inverse_transform(dataReconstructed)

    if args.post_cn:
        dataReduced = CenterScaler().transform(dataReduced)
        dataReduced = NormScaler().transform(dataReduced)

    val_l2 = rprec_a_l2(
        dataReduced["queries"],
        dataReduced["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        fast=True,
    )
    if args.post_cn:
        val_ip = val_l2
    else:
        val_ip = rprec_a_ip(
            dataReduced["queries"],
            dataReduced["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            fast=True,
        )
    loss_q = sklearn.metrics.mean_squared_error(
        data["queries"],
        dataReconstructed["queries"]
    )
    # loss of only the first 10k documents because it has to get copied
    loss_d = sklearn.metrics.mean_squared_error(
        data["docs"][:10000],
        dataReconstructed["docs"][:10000]
    )
    return val_ip, val_l2, loss_q, loss_d


def pca_performance_d(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
    ).fit(data_train["docs"])
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    return summary_performance(
        dataReduced,
        dataReconstructed
    )


def pca_performance_q(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
    ).fit(data_train["queries"])
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    return summary_performance(
        dataReduced,
        dataReconstructed
    )


def pca_performance_dq(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
        copy=False,
    ).fit(np.concatenate((data_train["queries"], data_train["docs"])))
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    return summary_performance(
        dataReduced,
        dataReconstructed
    )


DIMS = process_dims(args.dims)

# traverse from large to small
# DIMS.reverse()

logdata = []
for dim in DIMS:
    dim = int(dim)
    val_ip, val_l2, loss_q, loss_d = pca_performance_dq(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "dq"
    })
    val_ip, val_l2, loss_q, loss_d = pca_performance_d(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "d"
    })
    val_ip, val_l2, loss_q, loss_d = pca_performance_q(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "q"
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
