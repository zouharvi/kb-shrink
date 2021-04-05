#!/usr/bin/env python3

from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np

xtick_labels = [
    "768", "512", "512\ntanh", "256", "256\ntanh", "NECK",
    "NECK\ntanh", "256", "256\ntanh", "512", "512\ntanh", "768", "768\ntanh"
]
colors = list(mcolors.TABLEAU_COLORS.values())

data = defaultdict(lambda: defaultdict(list))
with open(sys.argv[1], 'r') as f:
    data_raw = [line.strip().split(",") for line in f.readlines()[1:]]
    for line in data_raw:
        data[line[0]]["index"].append(line[1])
        data[line[0]]["mrr_ip"].append(float(line[2]))
        data[line[0]]["mrr_l2"].append(float(line[3]))


fig = plt.figure(figsize=(8, 6))

# grey rectangles for tanh layers
for i, xtick_label in enumerate(xtick_labels):
    if "tanh" in xtick_label:
        plt.gca().add_patch(Rectangle(
            (i - 0.25, 0.1), 0.5, 0.925,
            facecolor='black',
            alpha=0.1,
            fill=True,
        ))

for i, (key, arrays) in enumerate(data.items()):
    # skip poitns for runs without activation
    data_local = [
        (index, mrr_ip, mrr_l2)
        for xtick_label, index, mrr_ip, mrr_l2
        in zip(xtick_labels, arrays["index"], arrays["mrr_ip"], arrays["mrr_l2"])
        if not ("tanh" in xtick_label and "no activ." in key)
    ]

    plt.plot(
        [x[0] for x in data_local],
        [x[1] for x in data_local],
        label=f"({key}) MRR IP",
        marker="^",
        alpha=0.5,
        color=colors[i],
    )
    plt.plot(
        [x[0] for x in data_local],
        [x[2] for x in data_local],
        label=f"({key}) MRR L2",
        marker="o",
        alpha=0.5,
        color=colors[i],
    )
plt.xlim(-0.25, 12.25)
plt.ylim(0.12, 1.06)

# text decorations
plt.gca().add_patch(Rectangle(
    (-1, 1.025), 14, 0.1,
    facecolor='black',
    alpha=0.1,
    fill=True,
))
plt.text(-0.17, 1.034, "Orig.")
plt.text(2.8, 1.034, "Encoder")
plt.text(9.1, 1.034, "Decoder")

plt.axvline(x=0.5, color='black', linestyle="dashed", alpha=0.3)
plt.axvline(x=6.5, color='black', linestyle="dashed", alpha=0.5)

plt.xticks(range(len(xtick_labels)), xtick_labels, rotation=0)
plt.xlabel("Layer (from which embedding is taken; bottleNECK size in parentheses)")
plt.ylabel("MRR (from layer embedding)")

plt.title("Embeddings of different autoencoder layers")
plt.legend(loc="lower left", ncol=2)
plt.tight_layout()
plt.show()
