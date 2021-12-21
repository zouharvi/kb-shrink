#!/usr/bin/env python3

raise NotImplementedError(
    "Not adapted to new data orgnization (docs and queries as tuples)")

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data
from misc.retrieval_utils import rprec_l2, rprec_ip
import numpy as np
import torch
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)

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
    avg_norm_q = np.average(
        torch.linalg.norm(torch.Tensor(dataReduced["queries"]), axis=1)
    )
    print(f"{name:<21} {loss_d:>7.5f} {loss_q:>7.5f} {val_ip:>5.3f} {val_l2:>5.3f}")
    return val_ip, val_l2, loss_q.item(), loss_d.item()


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
    return summary_performance(
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
    return summary_performance(
        f"PCA-Q ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_dq(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(np.concatenate((data["queries"], data["docs"])))
    dataReduced = {
        "queries": model.transform(data["queries"]),
        "docs": model.transform(data["docs"])
    }
    dataReconstructed = {
        "queries": model.inverse_transform(dataReduced["queries"]),
        "docs": model.inverse_transform(dataReduced["docs"])
    }
    return summary_performance(
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
    return summary_performance(
        f"Prec ({newType})",
        dataReduced,
        dataReconstructed
    )


# precision_performance("float32")
# precision_performance("float16")

data_log = []
for dim in np.linspace(32, 768, num=768 // 32, endpoint=True):
    # dim = 64
    dim = int(dim)
    val_ip, val_l2, loss_q, loss_d = pca_performance_d(dim)
    data_log.append({"type": "train_doc", "dim": dim, "val_ip": val_ip,
                    "val_l2": val_l2, "loss_q": loss_q, "loss_d": loss_d})
    val_ip, val_l2, loss_q, loss_d = pca_performance_q(dim)
    data_log.append({"type": "train_query", "dim": dim, "val_ip": val_ip,
                    "val_l2": val_l2, "loss_q": loss_q, "loss_d": loss_d})
    val_ip, val_l2, loss_q, loss_d = pca_performance_dq(dim)
    data_log.append({"type": "train_both", "dim": dim, "val_ip": val_ip,
                    "val_l2": val_l2, "loss_q": loss_q, "loss_d": loss_d})
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

# pca_precision_preformance(32, "float16")
# pca_precision_preformance(64, "float16")
# pca_precision_preformance(128, "float16")
# precision_pca_performance(128, "float16")
