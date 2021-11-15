#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
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
    data_small = dict(data)

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
    print(f"{name:<21} {loss_d:>7.5f} {loss_q:>7.5f} {val_ip:>5.3f} {val_l2:>5.3f}")
    return val_ip, val_l2, loss_q.item(), loss_d.item()

def safe_transform(model, array):
    return [model.transform([x])[0] for x in array]

def safe_inv_transform(model, array):
    return [model.inverse_transform([x])[0] for x in array]

def pca_performance_d(components):
    model = PCA(
        n_components=components,
        random_state=args.seed
    ).fit(data_small["docs"])
    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    dataReconstructed = {
        "queries": safe_inv_transform(model, dataReduced["queries"]),
        "docs": safe_inv_transform(model, dataReduced["docs"])
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
    ).fit(data_small["queries"])
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
    ).fit(np.concatenate((data_small["queries"], data_small["docs"])))
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


if args.dims == "custom":
    DIMS = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
elif args.dims == "linspace":
    DIMS = np.linspace(32, 768, num=768 // 32, endpoint=True)
else:
    raise Exception(f"Unknown --dims {args.dims} scheme")

# traverse from large to small
DIMS.reverse()

logdata = []
for dim in DIMS:
    dim = int(dim)
    val_ip, val_l2, loss_q, loss_d = pca_performance_dq(dim)
    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "dq"
    })
    val_ip, val_l2, loss_q, loss_d = pca_performance_d(dim)
    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "d"
    })
    val_ip, val_l2, loss_q, loss_d = pca_performance_q(dim)
    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "loss_q": loss_q, "loss_d": loss_d,
        "type": "q"
    })
    
    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
