#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/rproj.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())

DATA_GR = [x for x in DATA if x["model"] == "greedy"]
DATA = [x for x in DATA if x["model"] != "greedy"]


DIMS = sorted(list(set([x["dim"] for x in DATA])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.4, 3.6))
plt.rcParams["lines.linewidth"] = 2.2
ax = plt.gca()

ax.plot(
    [x["dim"] for x in DATA if x["model"] == "crop"],
    [max(x["vals_l2"]) for x in DATA if x["model"] == "crop"],
    label="Dim. Dropping", color="tab:red",
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "crop"],
    [min(x["vals_l2"]) for x in DATA if x["model"] == "crop"],
    alpha=0.4, color="tab:red", linestyle="-",
)

ax.plot(
    [768 - x["del_dim"] for x in DATA_GR],
    [x["val_l2"] for x in DATA_GR],
    label="Greedy Dim. Dropping", color="tab:red", linestyle=":",
)

ax.plot(
    [x["dim"] for x in DATA if x["model"] == "sparse"],
    [max(x["vals_l2"]) for x in DATA if x["model"] == "sparse"],
    label="Sparse", color="tab:blue", linestyle="-",
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "sparse"],
    [min(x["vals_l2"]) for x in DATA if x["model"] == "sparse"],
    alpha=0.4, color="tab:blue", linestyle="-",
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "gauss"],
    [max(x["vals_l2"]) for x in DATA if x["model"] == "gauss"],
    label="Gaussian", color="tab:blue", linestyle=":",
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "gauss"],
    [min(x["vals_l2"]) for x in DATA if x["model"] == "gauss"],
    alpha=0.4, color="tab:blue", linestyle=":",
)

# uncompressed
ax.axhline(y=0.618, alpha=0.5, linestyle="--", color="black")

plt.legend(
    ncol=1,
    loc="lower right",
)

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimensions")

plt.tight_layout()
plt.savefig("figures/random_projection.pdf")
plt.show()