#!/usr/bin/env python3

import copy
import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, process_dims, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import numpy as np
import torch
import argparse
import sklearn.metrics
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--data-small', default=None)
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--skip-loss', action="store_true")
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
print(f"{'Method':<21} {'Loss-D':<7} {'Loss-Q':<7} {'IPRPR':<0} {'L2RPR':<0}")


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
        fast=True, report=False
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
            fast=True, report=False
        )

    name = name.replace("float", "f")

    if not args.skip_loss:
        loss_q = sklearn.metrics.mean_squared_error(
            data["queries"],
            dataReconstructed["queries"]
        )
        # loss of only the first 10k documents because it has to get copied
        loss_d = sklearn.metrics.mean_squared_error(
            data["docs"][:10000],
            dataReconstructed["docs"][:10000]
        )
        print(f"{name:<21} {loss_d:>7.5f} {loss_q:>7.5f} {val_ip:>5.3f} {val_l2:>5.3f}")
        return val_ip, val_l2, loss_q.item(), loss_d.item()
    else:
        print(f"{name:<21} {-1:>7.5f} {-1:>7.5f} {val_ip:>5.3f} {val_l2:>5.3f}")
        return val_ip, val_l2, None, None


def safe_transform(model, array):
    return [model.transform([x])[0] for x in array]


def safe_inv_transform(model, array):
    return [model.inverse_transform([x])[0] for x in array]


def pca_performance_d(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data_small["docs"])
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

    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    if not args.skip_loss:
        dataReconstructed = {
            "queries": safe_inv_transform(model, dataReduced["queries"]),
            "docs": safe_inv_transform(model, dataReduced["docs"])
        }
    else:
        dataReconstructed = None
    return summary_performance(
        f"PCA-D ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_q(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data_small["queries"])
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

    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    if not args.skip_loss:
        dataReconstructed = {
            "queries": safe_inv_transform(model, dataReduced["queries"]),
            "docs": safe_inv_transform(model, dataReduced["docs"])
        }
    else:
        dataReconstructed = None
    return summary_performance(
        f"PCA-D ({components})",
        dataReduced,
        dataReconstructed
    )


def pca_performance_dq(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(np.concatenate((data_small["queries"], data_small["docs"])))
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

    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    if not args.skip_loss:
        dataReconstructed = {
            "queries": safe_inv_transform(model, dataReduced["queries"]),
            "docs": safe_inv_transform(model, dataReduced["docs"])
        }
    else:
        dataReconstructed = None
    return summary_performance(
        f"PCA-D ({components})",
        dataReduced,
        dataReconstructed
    )


DIMS = process_dims(args.dims)

logdata = []
for dim in DIMS:
    dim = int(dim)
    val_ip, val_l2, loss_q, loss_d = pca_performance_q(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "q"
    })
    val_ip, val_l2, loss_q, loss_d = pca_performance_d(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "d"
    })
    val_ip, val_l2, loss_q, loss_d = pca_performance_dq(dim)
    logdata.append({
        "dim": dim,
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "dq"
    })
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
