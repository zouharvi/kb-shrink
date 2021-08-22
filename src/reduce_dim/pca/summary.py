#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_keys_pickle, save_keys_pickle
import torch
import argparse
from sklearn.decomposition import PCA
from scipy.spatial.distance import minkowski
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PCA summary plots')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd")
parser.add_argument(
    '--keys-out-pca', default="data/eli5-dev-pca-256.embd")
args = parser.parse_args()

data = read_keys_pickle(args.keys_in)

# PCA reconstruction loss
reconstructionLosses = []
for components in [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 768]:
    model = PCA(n_components=components).fit(data)
    dataReduced = model.transform(data)
    dataDenoised = model.inverse_transform(dataReduced)

    loss = torch.nn.MSELoss()(torch.Tensor(data), torch.Tensor(dataDenoised))
    
    print(components, loss)
    reconstructionLosses.append((components, loss))

plt.figure(figsize=(6, 3.5))
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

# PCA dimensions
model = PCA(n_components=256).fit(data)
dataReduced = model.transform(data)
save_keys_pickle(dataReduced, args.keys_out_pca)
plt.figure(figsize=(6, 3.5))
plt.scatter(list(range(len(dataReduced[0])))*len(dataReduced), np.array(dataReduced).flatten(), alpha=0.2, marker='.', s=2)
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.title(f"Dimensions of {len(dataReduced)} keys/embeddings PCA (256)")
plt.tight_layout()
plt.show()

# PCA reconstructed dimensions
model = PCA(n_components=256).fit(data)
dataReconstructed = model.inverse_transform(model.transform(data))
plt.figure(figsize=(6, 3.5))
plt.scatter(list(range(len(dataReconstructed[0])))*len(dataReconstructed), np.array(dataReconstructed).flatten(), alpha=0.2, marker='.', s=2)
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.title(f"Dimensions of reconstructed {len(dataReconstructed)} keys/embeddings PCA (256)")
plt.tight_layout()
plt.show()

# save PCA
model = PCA(n_components=256).fit(data)
dataReduced = model.transform(data)
save_keys_pickle(dataReduced, args.keys_out_pca)