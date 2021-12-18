#!/usr/bin/env python3

import random
import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import numpy as np
import torch
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--data-big')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
data_big = read_pickle(args.data_big)

def summary_performance(name, dataReduced):
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
    return val_ip, val_l2


def pca_performance_d(components, data, data_train):
    model = PCA(
        n_components=components,
        random_state=args.seed
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
        f"PCA-D ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_q(components, data, data_train):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data_train["queries"])
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    return summary_performance(
        f"PCA-Q ({components})",
        dataReduced
    )

data_train = dict(data)
data = sub_data(data, train=False, in_place=True)
data_train = sub_data(data_train, train=True, in_place=True)
# data_big = sub_data(data_big, train=True, in_place=True)

print(len(data["docs"]), len(data["queries"]))
print(len(data_train["docs"]), len(data_train["queries"]))
print(len(data_big["docs"]), len(data_big["queries"]))

logdata = []

for num_samples in [10**3, (10**3)*5, (10**4)*5, 10**5, (10**5)*5, 10**6, (10**6)*5, 10**7, (10**7)*5]:

    new_data = dict(data)
    if num_samples < len(new_data["docs"]):
        new_data["docs"] = random.sample(new_data["docs"], num_samples)
    else:
        new_data["docs"] += random.sample(data_big["docs"], num_samples-len(new_data["docs"]))

    val_ip, val_l2 = pca_performance_d(args.dim, data, data_train)

    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "type": "q", 
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))


# val_ip, val_l2, loss_q, loss_d = pca_performance_d(args.dim)
# print({
#     "val_ip": val_ip, "val_l2": val_l2,
#     "loss_q": loss_q, "loss_d": loss_d,
#     "type": "d"
# })