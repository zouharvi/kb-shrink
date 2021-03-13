#!/usr/bin/env python3

import pickle
import argparse
import random
from sklearn.neighbors import KDTree, NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reduce KB key count')
parser.add_argument(
    '--prob', type=float, default=0.5,
    help='Probability by which to DISCARD every key')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument(
    '--metric', type=float, default=2,
    help='Parameter p for Minkowski distance metric')
args = parser.parse_args()
random.seed(args.seed)

data = []
with open(args.keys_in, "rb") as fread:
    reader = pickle.Unpickler(fread)
    while True:
        try:
            data.append(reader.load()[0])
        except EOFError:
            break

data = np.array(data)
model = NearestNeighbors(
    n_neighbors=2, algorithm="kd_tree",
    metric="minkowski", p=args.metric
).fit(data)
# the first retrieved neighbour is the vector itself
distancesOrig, indices = model.kneighbors(data)
distancesOrig = [x[1] for x in distancesOrig]

mask = np.ones(len(data), np.bool)
mask[[x[1] for x in indices]] = 0
data = data[mask]
model = NearestNeighbors(
    n_neighbors=2, algorithm="kd_tree",
    metric="minkowski", p=args.metric
).fit(data)
# the first retrieved neighbour is the vector itself
distancesNew, indices = model.kneighbors(data)
distancesNew = [x[1] for x in distancesNew]


_, bins, _ = plt.hist(
    [distancesOrig, distancesNew], bins=10,
    label=["Original", "Nearest neighbours removed"])
plt.xticks(bins, [f'{x:.2f}' for x in bins], rotation=45)
plt.xlabel(f"Minkowski distance $p={args.metric}$")
plt.ylabel("Bucket count")
plt.title(args.keys_in.split("/")[-1] + " histogram of nearest neighbour distances")
plt.tight_layout()
plt.legend()
plt.show()
