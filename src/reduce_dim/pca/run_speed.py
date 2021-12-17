#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data
import numpy as np
import argparse
from sklearn.decomposition import PCA
from reduce_dim.pca.model_torch import TorchPCA
import time

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

if args.pca_model == "scikit":
    ModelClass = PCA
elif args.pca_model == "torch":
    ModelClass = TorchPCA
else:
    raise Exception("Unknown PCA model selected")

def safe_transform(model, array):
    return [model.transform([x])[0] for x in array]

def safe_inv_transform(model, array):
    return [model.inverse_transform([x])[0] for x in array]

def pca_performance_dq(components):
    train_time = time.time()
    model = ModelClass(
        n_components=components,
        random_state=args.seed,
        copy=False,
    ).fit(np.concatenate((data_train["queries"], data_train["docs"])))
    train_time = train_time - time.time()

    encode_time = time.time()
    dataReduced = {
        "queries": safe_transform(model, data["queries"]),
        "docs": safe_transform(model, data["docs"])
    }
    encode_time = encode_time - time.time()

DIMS = process_dims(args.dims)

logdata = []
for dim in DIMS:
    dim = int(dim)
    train_time, encode_time = pca_performance_dq(dim)
    logdata.append({"dim": dim, "train_time": train_time, "encode_time": encode_time, "type": "dq"})

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
