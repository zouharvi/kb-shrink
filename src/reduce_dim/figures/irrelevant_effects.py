#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--logfile-pca', default="computed/pca_irrelevant_uneasiness.log")
parser.add_argument(
    '--logfile-auto', default="computed/auto_irrelevant_uneasiness.log")
parser.add_argument('--key', default=0, type=int)
args = parser.parse_args()

with open(args.logfile_pca, "r") as f:
    data_pca = eval(f.read())
with open(args.logfile_auto, "r") as f:
    data_auto = eval(f.read())

DIMS = [
    np.log10(x["num_samples"])
    for x in data_pca if x["type"] == "train_data"
]
DIMS_NEXT = [
    np.log10(x["num_samples"])
    for x in data_pca if x["type"] == "eval_data"
]
DISPLAY_DIMS = [128, 10**3, (10**4), 10**5, 10**6, 10**7, (10**7) * 3]
THRESHOLD = 2114151

plt.figure(figsize=(4.6, 3.6))
plt.rcParams["lines.linewidth"] = 2.2
ax = plt.gca()

ax.plot(
    DIMS,
    [x["val_ip"] for x in data_pca if x["type"] == "train_data"],
    label="PCA (training docs)", color="tab:blue", linestyle="-")
ax.plot(
    DIMS_NEXT,
    [x["val_ip"] for x in data_pca if x["type"] == "eval_data"],
    label="PCA (eval docs)", color="tab:blue", linestyle=":")
ax.plot(
    DIMS,
    [x["val_ip"] for x in data_auto if x["type"] == "train_data"],
    label="Auto. (training docs)", color="tab:red", linestyle="-")
ax.plot(
    DIMS_NEXT,
    [x["val_ip"] for x in data_auto if x["type"] == "eval_data"],
    label="Auto. (eval docs)", color="tab:red", linestyle=":")

ax.set_xticks([np.log10(x) for x in DISPLAY_DIMS])
ax.set_xticklabels([
    x if x == 128 else '$10^{' + f'{np.log10(x):.1f}' + '}$'
    for x in DISPLAY_DIMS
])
ax.set_ylabel("R-Precision")
ax.set_xlabel("Docs count (log scale)")
ax.set_ylim(0.37, 0.62)
ax.set_xlim(2, 7.6)

plt.scatter(
    np.log10(THRESHOLD),
    [
        x["val_ip"]
        for x in data_pca
        if x["type"] == "train_data" and x["num_samples"] == THRESHOLD
    ],
    marker="x", color="black", zorder=10, s=17,
)
plt.scatter(
    np.log10(THRESHOLD),
    [
        x["val_ip"]
        for x in data_auto
        if x["type"] == "train_data" and x["num_samples"] == THRESHOLD
    ],
    marker="x", color="black", zorder=10, s=17
)

h1, l1 = ax.get_legend_handles_labels()

# plt.title(["No pre-processing", "Normalized", "Centered", "Centered, Normalized"][args.key])
plt.legend(
    h1, l1,
    loc="center",
    bbox_to_anchor=(0, 0.99, 1, 0.28),
    ncol=2
)
plt.tight_layout(rect=(0, 0, 1, 1.03))
plt.savefig("figures/model_data.pdf")
plt.show()
