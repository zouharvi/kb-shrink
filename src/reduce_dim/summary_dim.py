#!/usr/bin/env python3

import pickle
import argparse
import random
from sklearn.decomposition import PCA
from scipy.spatial.distance import minkowski
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("src")
from misc.utils import read_keys_pickle

parser = argparse.ArgumentParser(description='Explore vector distribution')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
args = parser.parse_args()

data = read_keys_pickle(args.keys_in)

# PCA reconstruction loss
reconstructionLosses = []
for components in [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 768]:
    model = PCA(n_components=components).fit(data)
    dataReduced = model.transform(data)
    dataDenoised = model.inverse_transform(dataReduced)

    distances = []
    for vec, vecDenoised in zip(data, dataDenoised):
        distances.append(minkowski(vec, vecDenoised, 2))
    
    print(components, np.average(distances))
    reconstructionLosses.append((components, np.average(distances)))

plt.plot([x[0] for x in reconstructionLosses], [x[1] for x in reconstructionLosses], marker="o")
plt.title(f"Average L2 PCA reconstruction loss for {len(data)} keys/embeddings")
plt.xlabel("PCA components")
plt.ylabel("Average L2 reconstruction loss")
plt.tight_layout()
plt.show()

# Original dimensions
plt.figure(figsize=(6,3.5))
plt.scatter(list(range(len(data[0])))*len(data), np.array(data).flatten(), alpha=0.2, marker='.', s=2)
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.title(f"Dimensions of {len(data)} keys/embeddings")
plt.tight_layout()
plt.show()

# PCA reconstructed dimensions
model = PCA(n_components=256).fit(data)
dataReduced = model.inverse_transform(model.transform(data))
plt.figure(figsize=(6, 3.5))
plt.scatter(list(range(len(dataReduced[0])))*len(dataReduced), np.array(dataReduced).flatten(), alpha=0.2, marker='.', s=2)
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.title(f"Dimensions of {len(dataReduced)} keys/embeddings reduced PCA (256)")
plt.tight_layout()
plt.show()