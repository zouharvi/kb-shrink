#!/usr/bin/env python3

import pickle
import argparse
import random
from sklearn.neighbors import KDTree, NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reduce KB key count')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
parser.add_argument(
    '--metric', type=float, default=2,
    help='Parameter p for Minkowski distance metric')
args = parser.parse_args()

data = []
with open(args.keys_in, "rb") as fread:
    reader = pickle.Unpickler(fread)
    while True:
        try:
            data.append(reader.load()[0])
        except EOFError:
            break
data = np.array(data)

# original
model = NearestNeighbors(
    n_neighbors=2, algorithm="kd_tree",
    metric="minkowski", p=args.metric
).fit(data)
distancesOrig, indices = model.kneighbors(data)
distancesOrig = [x[1] for x in distancesOrig]

# DATA DEL
mask = np.ones(len(data), np.bool)
mask[[x[1] for x in indices]] = 0
dataDelN = data[mask]
model = NearestNeighbors(
    n_neighbors=2, algorithm="kd_tree",
    metric="minkowski", p=args.metric
).fit(dataDelN)
distancesDel, _ = model.kneighbors(dataDelN)
distancesDel = [x[1] for x in distancesDel]

# DATA SYM
dataSym = []
indicesSym = indices.copy()
for cur_index, (_, tar_index) in enumerate(indicesSym):
    # if not removed by clustering
    if tar_index != -1 and indicesSym[tar_index][1] != -1:
        if indices[tar_index][1] == cur_index:
            dataSym.append((data[cur_index] + data[tar_index])/2)
            indicesSym[tar_index] = -1
        else:
            dataSym.append(data[cur_index])
dataSym = np.array(dataSym)
model = NearestNeighbors(
    n_neighbors=2, algorithm="kd_tree",
    metric="minkowski", p=args.metric
).fit(dataSym)
distancesSym, _ = model.kneighbors(dataSym)
distancesSym = [x[1] for x in distancesSym]

# DATA GREEDY
dataGreed = []
indicesGreed = indices.copy()
for cur_index, (_, tar_index) in enumerate(indicesGreed):
    # if not removed by clustering
    if tar_index != -1 and indicesGreed[tar_index][1] != -1:
        dataGreed.append((data[cur_index] + data[tar_index])/2)
        indicesGreed[tar_index] = -1
dataGreed = np.array(dataGreed)
model = NearestNeighbors(
    n_neighbors=2, algorithm="kd_tree",
    metric="minkowski", p=args.metric
).fit(dataGreed)
distancesGreed, _ = model.kneighbors(dataGreed)
distancesGreed = [x[1] for x in distancesGreed]

histData = [distancesOrig, distancesDel, distancesGreed, distancesSym]
_, bins, _ = plt.hist(
    histData,
    weights=[np.ones(len(arr))/len(arr) for arr in histData],
    bins=7,
    label=[
        "Original",
        f"Nearest neighbors removed\nDeleted: {len(distancesOrig)-len(distancesDel)}",
        f"Greedy neighbors averaged\nDeleted: {len(distancesOrig)-len(distancesGreed)}",
        f"Symmetric neighbors averaged\nDeleted: {len(distancesOrig)-len(distancesSym)}"
])
plt.xticks(bins, [f'{x:.2f}' for x in bins], rotation=45)
plt.xlabel(f"Minkowski distance $p={args.metric}$")
plt.ylabel("Bucket count")
plt.title(args.keys_in.split("/")[-1] + " histogram of nearest neighbour distances")
plt.tight_layout()
plt.legend()
plt.show()
