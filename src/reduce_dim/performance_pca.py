#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_pickle, acc_l2, acc_ip
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--keys', default="data/hotpot.embd")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.keys)

print(f"{'Method':<21} {'Loss-D':<7} {'Loss-Q':<7} {'IPACC':<0} {'L2ACC':<0}")


def summary_performance(name, dataReduced, dataReconstructed):
    acc_val_ip = acc_ip(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], 20
    )
    acc_val_l2 = acc_l2(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], 20
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
    avg_norm_q = np.average(
        torch.linalg.norm(torch.Tensor(dataReduced["queries"]), axis=1)
    )
    print(f"{name:<21} {loss_d:>7.5f} {loss_q:>7.5f} {acc_val_ip:>5.3f} {acc_val_l2:>5.3f}")


def pca_performance_d(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data["docs"])
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    summary_performance(
        f"PCA-D ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_q(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data["queries"])
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    summary_performance(
        f"PCA-Q ({components})",
        dataReduced,
        dataReconstructed
    )

def pca_performance_dq(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data["queries"]+data["docs"])
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    summary_performance(
        f"PCA-DQ ({components})",
        dataReduced,
        dataReconstructed
    )


def precision_performance(newType):
    dataReduced = {
        "queries": np.array(data["queries"], dtype=newType),
        "docs": np.array(data["docs"], dtype=newType)
    }
    dataReconstructed = {
        "queries": dataReduced["queries"].astype("float32"),
        "docs": dataReduced["docs"].astype("float32")
    }
    summary_performance(
        f"Prec ({newType})",
        dataReduced,
        dataReconstructed
    )


# precision_performance("float32")
# precision_performance("float16")
# pca_performance_d(64)
# pca_performance_q(64)
# pca_performance_dq(64)
# pca_performance_d(128)
# pca_performance_q(128)
pca_performance_dq(128)
# pca_precision_preformance(32, "float16")
# pca_precision_preformance(64, "float16")
# pca_precision_preformance(128, "float16")
# precision_pca_performance(128, "float16")
