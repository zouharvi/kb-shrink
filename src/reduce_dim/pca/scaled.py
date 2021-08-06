#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

print(f"{'Method':<21} {'Loss-D':<7} {'Loss-Q':<7} {'IPRPR':<0} {'L2RPR':<0}")

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
    name = name.replace("float", "f")
    print(f"{name:<21} {loss_d:>7.5f} {loss_q:>7.5f} {val_ip:>5.3f} {val_l2:>5.3f}")
    return val_ip, val_l2, loss_q.item(), loss_d.item()

def pca_performance_d(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data["docs"])
    # model.components_ = model.components_[1:]
    eigenvalues = model.explained_variance_
    scaling = np.ones(components)
    # IP optimized
    # scaling[0] = 0.5
    # scaling[1] = 0.8
    # scaling[2] = 0.8
    # scaling[3] = 0.9
    # scaling[4] = 0.8
    # L2 optimized
    scaling[0] = 0.5
    scaling[1] = 0.8
    scaling[2] = 0.7
    scaling[3] = 0.9
    scaling[4] = 0.6
    model.components_ *= scaling[:, np.newaxis]
    
    print(eigenvalues[:6])

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


# data_log = []
val_ip, val_l2, loss_q, loss_d = pca_performance_d(128)
# data_log.append({"type": "train_doc", "dim": dim, "val_ip": val_ip, "val_l2": val_l2, "loss_q": loss_q, "loss_d": loss_d})
    # # continuously override the file
    # with open(args.logfile, "w") as f:
    #     f.write(str(data_log))