#!/usr/bin/env python3

import sys
sys.path.append("../../")
from misc.load_utils import read_pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

plt.figure(figsize=(10,7))
data = read_pickle("/data/kilt/hotpot-dpr-c-5000.embd")
tsne = TSNE(2, random_state=0)
tsne = tsne.fit_transform(np.array(data["queries"]+data["docs"][:5000]))
plt.scatter(tsne[:5000,0], tsne[:5000,1], alpha=0.3, s=5)
plt.scatter(tsne[5000:,0], tsne[5000:,1], alpha=0.3, s=5, color="red")
plt.title(f"t-SNE visualization")

plt.tight_layout()
plt.show()