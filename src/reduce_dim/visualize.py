#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_keys_pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='Visualization of embeddings')
parser.add_argument(
    '--keys-1', default="data/eli5-dev.embd")
parser.add_argument(
    '--keys-2', default="data/eli5-dev-pca-256.embd")
parser.add_argument(
    '--keys-3', default="data/eli5-dev-auto-256.embd")
args = parser.parse_args()

plt.figure(figsize=(15,4))

data = read_keys_pickle(args.keys_1)
tsne = TSNE(2, random_state=0)
tsne = tsne.fit_transform(data)
plt.subplot(1, 3, 1)
plt.scatter(tsne[:,0], tsne[:,1], alpha=0.3, s=5)
plt.title(f"t-SNE visualization\n({args.keys_1})")

data = read_keys_pickle(args.keys_2)
tsne = TSNE(2, random_state=0)
tsne = tsne.fit_transform(data)
plt.subplot(1, 3, 2)
plt.scatter(tsne[:,0], tsne[:,1], alpha=0.3, s=5, color="tab:orange")
plt.title(f"t-SNE visualization\n({args.keys_2})")

data = read_keys_pickle(args.keys_3)
tsne = TSNE(2, random_state=0)
tsne = tsne.fit_transform(data)
plt.subplot(1, 3, 3)
plt.scatter(tsne[:,0], tsne[:,1], alpha=0.3, s=5, color="tab:green")
plt.title(f"t-SNE visualization\n({args.keys_3})")

plt.tight_layout()
plt.show()