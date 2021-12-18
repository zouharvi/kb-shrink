#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile-pca', default="computed/pca_irrelevant_uneasiness.log")
parser.add_argument('--logfile-auto', default="computed/auto_irrelevant_uneasiness.log")
parser.add_argument('--key', default=0, type=int)
args = parser.parse_args()

with open(args.logfile_pca, "r") as f:
    data_pca = eval(f.read())
# with open(args.logfile_auto, "r") as f:
#     data_auto = eval(f.read())

DIMS = [np.log10(x["num_samples"]) for x in data_pca if x["type"] == "train_data"]
DIMS_NEXT = [np.log10(x["num_samples"]) for x in data_pca if x["type"] == "eval_data"]
DISPLAY_DIMS = ['$10^{' + f'{np.log10(x["num_samples"]):.1f}' + '}$' for x in data_pca if x["type"] == "train_data"]
# DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.8, 4.7))
ax = plt.gca() 

ax.plot(DIMS, [x["val_ip"] for x in data_pca if x["type"] == "train_data"], label="PCA", color="tab:blue", linestyle="-")
ax.plot(DIMS_NEXT, [x["val_ip"] for x in data_pca if x["type"] == "eval_data"], label="PCA", color="tab:blue", linestyle=":")

ax.set_xticks(DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Prec")
ax.set_xlabel("Sample size")
# ax.set_ylim(0.015, 0.66)

h1, l1 = ax.get_legend_handles_labels()

# plt.title(["No pre-processing", "Normalized", "Centered", "Centered, Normalized"][args.key])
plt.legend(
    h1, l1,
    loc="center",
    bbox_to_anchor=(-0.05, 1.12, 1, 0.2),
    ncol=2
)
plt.tight_layout(rect=(0, 0, 1, 1.05))
plt.show()