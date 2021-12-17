#!/usr/bin/env python3

import time
import torch
from model_torch import TorchPCA
from sklearn.decomposition import PCA
import argparse
import numpy as np
import sys; sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--pca-model', default="scikit")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--dims', default="custom")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)
print("Because args.data_small is not provided, I'm copying the whole structure")
data_train = dict(data)

data = sub_data(data, train=False, in_place=True)
data_train = sub_data(data_train, train=True, in_place=True)

def safe_transform(model, array):
    dataLoader = torch.utils.data.DataLoader(
        dataset=array, batch_size=1024*128, shuffle=False
    )

    out = []
    for sample in dataLoader:
        out.append(model.transform(sample))
    return out

def pca_performance_dq(components):
    train_time = time.time()
    if args.pca_model == "scikit":
        model = PCA(
            n_components=components,
            random_state=args.seed,
            copy=False,
        ).fit(np.concatenate((data_train["queries"], data_train["docs"])))
    elif args.pca_model == "torch":
        model = TorchPCA(
            n_components=components,
        ).fit(np.concatenate((data_train["queries"], data_train["docs"])))
    else:
        raise Exception("Unknown PCA model selected")
    
    train_time = time.time()-train_time

    encode_time = time.time()
    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    encode_time = time.time()-encode_time
    return train_time, encode_time


DIMS = process_dims(args.dims)

logdata = []
for dim in DIMS:
    dim = int(dim)
    train_time, encode_time = pca_performance_dq(dim)
    logdata.append({
        "dim": dim, "train_time": train_time,
        "encode_time": encode_time, "type": "dq"}
    )

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
