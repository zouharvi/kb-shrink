#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data, order_ip, order_l2
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

model = PCA(
    n_components=64,
    random_state=args.seed
).fit(np.concatenate((data["queries"], data["docs"])))

print("Q * Q^T")
print((np.matmul(model.components_, model.components_.T) >= 0.001)*1)

print("Q^T * Q")
print((np.matmul(model.components_.T, model.components_) >= 0.001)*1)

dataReduced = {
    "queries": model.transform(data["queries"]),
    "docs": model.transform(data["docs"])
}


print("\nReduced norms")
print(f"Queries: {np.average(np.linalg.norm(dataReduced['queries'], axis=1)[:, np.newaxis]):.8f}")
print(f"Docs:    {np.average(np.linalg.norm(dataReduced['docs'], axis=1)[:, np.newaxis]):.8f}")

print("\nPerformance")
val_ip_pca = rprec_ip(
    dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
)
val_l2_pca = rprec_l2(
    dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
)
print(f"IP: {val_ip_pca}")
print(f"L2: {val_l2_pca}")

print("\nRenormalized performance")
dataReduced = center_data(dataReduced)
dataReduced = norm_data(dataReduced)
val_ip_pca = rprec_ip(
    dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
)
val_l2_pca = rprec_l2(
    dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
)
print(f"IP: {val_ip_pca}")
print(f"L2: {val_l2_pca}")

print("\nOverlap of retrievals")
val_order_l2 = list(order_l2(
    dataReduced["queries"], dataReduced["docs"],
    [len(x) for x in data["relevancy"]],
    fast=True
))
val_order_ip = list(order_ip(
    dataReduced["queries"], dataReduced["docs"],
    [len(x) for x in data["relevancy"]],
    fast=True
))
print(f"Strict equality    [x==y]:     {np.average([set(x[:len(z)]) == set(y[:len(z)]) for x,y,z in zip(val_order_l2, val_order_ip, data['relevancy'])])*100:.2f}%")
print(f"Partial equality   [|x&y|!=0]: {np.average([len(set(x[:len(z)]) & set(y[:len(z)])) != 0 for x,y,z in zip(val_order_l2, val_order_ip,data['relevancy'])])*100:.2f}%")
print(f"Average overlap    [|x&y|]:    {np.average([len(set(x[:len(z)]) & set(y[:len(z)])) for x,y,z in zip(val_order_l2, val_order_ip,data['relevancy'])]):.2f}")
print(f"Average retrieved  [|x|]:      {np.average([len(x) for x in data['relevancy']]):.2f}")