#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_keys_pickle, acc_l2_fast, acc_ip_fast
import argparse
from sklearn.decomposition import PCA
from pympler.asizeof import asizeof

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--keys', default="data/eli5-dev.embd")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_keys_pickle(args.keys)
origSize = asizeof(data)

print(data.shape)

print(f"{'Method':<21} {'Size':<6} {'L2 Loss':<7} {'IPACC':<0} {'L2ACC':<0} {'norm':<0}")


def summary_performance(name, dataReduced, dataReconstructed):
    acc_val_ip = acc_ip_fast(data, dataReduced, 20, report=False)
    acc_val_l2 = acc_l2_fast(data, dataReduced, 20, report=False)
    size = asizeof(dataReduced)
    loss = torch.nn.MSELoss()(torch.Tensor(data), torch.Tensor(dataReconstructed))
    name = name.replace("float", "f")
    avg_norm = np.average(torch.linalg.norm(torch.Tensor(dataReduced), axis=1))
    print(f"{name:<21} {size/origSize:>5.3f}x {loss:>7.5f} {acc_val_ip:>5.3f} {acc_val_l2:>5.3f} {avg_norm:>4.2f}")


def pca_performance(components):
    model = PCA(n_components=components, random_state=args.seed).fit(data)
    dataReduced = model.transform(data)
    summary_performance(
        f"PCA ({components})",
        dataReduced,
        model.inverse_transform(dataReduced)
    )


def precision_performance(newType):
    dataReduced = data.astype(newType)
    summary_performance(
        f"Prec ({newType})",
        dataReduced,
        dataReduced.astype("float32")
    )


def precision_pca_performance(components, newType):
    dataReduced = data.astype(newType)
    model = PCA(n_components=components,
                random_state=args.seed).fit(dataReduced)
    dataReduced = model.transform(dataReduced).astype("float32")
    summary_performance(
        f"Prec ({newType}), PCA ({components})",
        dataReduced,
        model.inverse_transform(dataReduced).astype("float32")
    )


def pca_precision_preformance(components, newType):
    model = PCA(n_components=components, random_state=args.seed).fit(data)
    dataReduced = model.transform(data)
    dataReduced = dataReduced.astype(newType)
    summary_performance(
        f"PCA ({components}), Prec ({newType})",
        dataReduced,
        model.inverse_transform(dataReduced.astype("float32"))
    )


summary_performance(f"Original ({data.dtype})", data, data)
pca_performance(32)
pca_performance(64)
pca_performance(128)
pca_performance(256)
precision_performance("float16")
pca_precision_preformance(32, "float16")
pca_precision_preformance(64, "float16")
pca_precision_preformance(128, "float16")
precision_pca_performance(128, "float16")
