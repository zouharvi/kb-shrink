#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data
from misc.retrieval_utils import rprec_l2, rprec_ip
import numpy as np
import torch
import argparse
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

def summary_performance(name, dataReduced, dataReconstructed):
    if args.post_cn:
        dataReduced = center_data(dataReduced)
        dataReduced = norm_data(dataReduced)

    val_ip = rprec_ip(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    val_l2 = rprec_l2(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    loss_q = torch.nn.MSELoss()(
        torch.Tensor(data["queries"]),
        torch.Tensor(dataReconstructed["queries"])
    )
    loss_d = torch.nn.MSELoss()(
        torch.Tensor(data["docs"]),
        torch.Tensor(dataReconstructed["docs"])
    )
    return val_ip, val_l2, loss_q.item(), loss_d.item()

def pca_performance_d(params, components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data["docs"])
    
    scaling = np.ones(components)
    for k,v in params.items():
        scaling[k] = v
    model.components_ *= scaling[:, np.newaxis]
    
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


param_grid = {}
for i in range(5):
    param_grid[i] = np.linspace(0.5,1,num=6,endpoint=True)
param_grid = ParameterGrid(param_grid)

def argmax(iterable, key="val_ip"):
    return max(enumerate(iterable), key=lambda x: x[1][key])[1]

data_log = []
for params in param_grid:
    val_ip, val_l2, loss_q, loss_d = pca_performance_d(params, 128)
    data_log.append({"params": params, "val_ip": val_ip, "val_l2": val_l2})
    print("-----")
    print({"params": params, "val_ip": val_ip, "val_l2": val_l2})
    print("current best-IP", argmax(data_log, "val_ip"))
    print("current best-L2", argmax(data_log, "val_l2"))
    print()


print(argmax(data_log, "val_ip"))
print(argmax(data_log, "val_l2"))