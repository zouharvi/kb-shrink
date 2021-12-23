#!/usr/bin/env python3

import copy
import sys
sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import numpy as np
import argparse
from sklearn.decomposition import PCA
import sklearn.metrics

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--data-small', default=None)
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--dims', default="custom")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

if args.data_small is None:
    if args.center:
        data = center_data(data)
    if args.norm:
        data = norm_data(data)
    print("Because args.data_small is not provided, I'm copying the whole structure")
    data_small = copy.deepcopy(data)

    data = sub_data(data, train=False, in_place=True)
    data_small = sub_data(data_small, train=True, in_place=True)

else:
    data_small = read_pickle(args.data_small)
    if args.center:
        data = center_data(data)
        data_small = center_data(data_small)
    if args.norm:
        data = norm_data(data)
        data_small = norm_data(data_small)
    data = sub_data(data, train=False, in_place=True)
    data_small = sub_data(data_small, train=True, in_place=True)


def safe_print(msg):
    with open("base_big_pca.out", "a") as f:
        f.write(msg + "\n")


def summary_performance(name, dataReduced, dataReconstructed):
    if args.post_cn:
        dataReduced = center_data(dataReduced)
        dataReduced = norm_data(dataReduced)

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
    name = name.replace("float", "f")
    return val_ip, val_l2, loss_q, loss_d


def safe_transform(model, array):
    return [model.transform([x])[0] for x in array]


def safe_inv_transform(model, array):
    return [model.inverse_transform([x])[0] for x in array]


def pca_performance_d(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
    ).fit(data_small["docs"])
    safe_print("Ed")
    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    safe_print("Fd")
    dataReconstructed = {
        "queries": safe_inv_transform(model, dataReduced["queries"]),
        "docs": safe_inv_transform(model, dataReduced["docs"])
    }
    safe_print("Gd")
    return summary_performance(
        f"PCA-D ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_q(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
    ).fit(data_small["queries"])
    safe_print("Eq")
    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    safe_print("Fq")
    dataReconstructed = {
        "queries": safe_inv_transform(model, dataReduced["queries"]),
        "docs": safe_inv_transform(model, dataReduced["docs"])
    }
    safe_print("Gq")
    return summary_performance(
        f"PCA-Q ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_dq(components):
    model = PCA(
        n_components=components,
        random_state=args.seed,
        copy=False,
    ).fit(np.concatenate((data_small["queries"], data_small["docs"])))
    safe_print("Edq")
    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    safe_print("Fdq")
    dataReconstructed = {
        "queries": safe_inv_transform(model, dataReduced["queries"]),
        "docs": safe_inv_transform(model, dataReduced["docs"])
    }
    safe_print("Gdq")
    return summary_performance(
        f"PCA-DQ ({components})",
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
    safe_print("-")
    val_ip, val_l2, loss_q, loss_d = pca_performance_d(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "d"
    })
    safe_print("-")
    val_ip, val_l2, loss_q, loss_d = pca_performance_q(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "q"
    })
    safe_print("-")

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))