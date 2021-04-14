#!/usr/bin/env python3

import sys, os

import torch
sys.path.append("src")
from misc.utils import mrr, read_keys_pickle, vec_sim_order, l2_sim
import argparse
from sklearn.decomposition import PCA
import numpy as np
from pympler.asizeof import asizeof

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--keys-in', default="data/eli5-dev.embd")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_keys_pickle(args.keys_in)
origSize = asizeof(data)

order_old_l2 = vec_sim_order(data, sim_func=l2_sim)
order_old_ip = vec_sim_order(data, sim_func=np.inner)

print(f"{'Method':<21} {'Size':<6} {'L2 Loss':<7} {'IPMRR':<0} {'L2MRR':<0}")

def summary_performance(name, dataReduced, dataReconstructed):
    order_new_ip = vec_sim_order(dataReduced, sim_func=np.inner)
    order_new_l2 = vec_sim_order(dataReduced, sim_func=l2_sim)
    mrr_val_ip = mrr(order_old_ip, order_new_ip, 20, report=False)
    mrr_val_l2 = mrr(order_old_l2, order_new_l2, 20, report=False)
    size = asizeof(dataReduced)
    loss = torch.nn.MSELoss()(torch.Tensor(data), torch.Tensor(dataReconstructed))
    name = name.replace("float", "f")
    print(f"{name:<21} {size/origSize:>5.3f}x {loss:>7.5f} {mrr_val_ip:>5.3f} {mrr_val_l2:>5.3f}")

def pca_performance(components):
    model = PCA(n_components=components, random_state=args.seed).fit(data)
    dataReduced = model.transform(data)
    summary_performance(f"PCA ({components})", dataReduced, model.inverse_transform(dataReduced))

def precision_performance(newType):
    dataReduced = data.astype(newType)
    summary_performance(f"Prec ({newType})", dataReduced, dataReduced.astype("float32"))

def precision_pca_performance(components, newType):
    dataReduced = data.astype(newType)
    model = PCA(n_components=components, random_state=args.seed).fit(dataReduced)
    dataReduced = model.transform(dataReduced).astype("float32")
    summary_performance(f"Prec ({newType}), PCA ({components})", dataReduced, model.inverse_transform(dataReduced).astype("float32"))

def pca_precision_preformance(components, newType):
    model = PCA(n_components=components, random_state=args.seed).fit(data)
    dataReduced = model.transform(data)
    dataReduced = dataReduced.astype(newType)
    summary_performance(f"PCA ({components}), Prec ({newType})", dataReduced, model.inverse_transform(dataReduced.astype("float32")))

summary_performance(f"Original ({data.dtype})", data, data)
pca_performance(32)
pca_performance(64)
pca_performance(128)
pca_performance(256)
precision_performance("float16")
pca_precision_preformance(32, "float16")
pca_precision_preformance(64, "float16")
pca_precision_preformance(128, "float16")
precision_pca_performance(256, "float16")