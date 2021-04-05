#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np

with open(sys.argv[1], 'r') as f:
    data = [line.strip().replace(",", "").split(" ") for line in f.readlines()]
    data_names = [x[1].replace("_tanh", "\ntanh") for x in data]
    data_mrr_ip = [float(x[5]) for x in data]
    data_mrr_l2 = [float(x[7]) for x in data]


fig = plt.figure(figsize=(8,6))

plt.scatter(
    range(len(data_names)),
    data_mrr_ip,
    label="MRR IP"
)
plt.scatter(
    range(len(data_names)),
    data_mrr_l2,
    label="MRR L2",
    marker="^"
)
plt.axvline(x = 0.5, color = 'black', linestyle="dashed", alpha=0.3)
plt.axvline(x = 6.5, color = 'black', linestyle="dashed", alpha=0.5)

plt.xticks(range(len(data_names)), data_names, rotation=0)
plt.xlabel("Layer")
plt.ylabel("MRR")

plt.title("Embeddings of different autoencoder layers")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
