#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.retrieval_utils import DEVICE
import torch
from reduce_dim.autoencoder.model import AutoencoderModel
import numpy as np
import timeit
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', action="store_true")
parser.add_argument('--n', type=int, default=10000)
args = parser.parse_args()

data2transform = torch.rand(args.n, 768)
print("will transform", len(data2transform), "vectors")
data2transform = torch.Tensor(data2transform)

# scikit implementation is slower
# print("fitting PCA")
# model = PCA(
#     n_components=128,
#     random_state=args.seed
# ).fit(data["docs"])
# pca_time = timeit.timeit(lambda: model.transform(data2transform), number=10)
# print(f"PCA {pca_time:.2f}s per the whole batch")
# print(f"PCA {pca_time/len(data2transform)*1000000:.2f}µs per vector")

matU, matS, matV = torch.pca_lowrank(data2transform)

if args.gpu:
    matV = matV.to(DEVICE)
    data2transform = data2transform.to(DEVICE)

pca_time = timeit.timeit(lambda: torch.matmul(data2transform, matV[:, :128]), number=10)
print(f"PCA {pca_time:.2f}s per the whole batch")
print(f"PCA {pca_time/len(data2transform)*1000000:.4f}µs per vector")

# DEVICE = torch.device("cpu")
model = AutoencoderModel(model=1, bottleneck_width=128, skip_move=not args.gpu)
asingle_time = timeit.timeit(lambda: model.encode(data2transform), number=10)
print(f"ASingle {asingle_time:.2f}s per the whole batch")
print(f"ASingle {asingle_time/len(data2transform)*1000000:.4f}µs per vector")

model = AutoencoderModel(model=3, bottleneck_width=128, skip_move=not args.gpu)
ashallow_time = timeit.timeit(lambda: model.encode(data2transform), number=10)
print(f"AShallow {ashallow_time:.2f}s per the whole batch")
print(f"AShallow {ashallow_time/len(data2transform)*1000000:.4f}µs per vector")