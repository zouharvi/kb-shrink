#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/rproj.py")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())

DATA_GR = [x for x in DATA if x["model"] == "greedy"]
DATA = [x for x in DATA if x["model"] != "greedy"]


DIMS = sorted(list(set([x["dim"] for x in DATA])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.8, 3.8))
ax = plt.gca()

ax.plot(
    [x["dim"] for x in DATA if x["model"] == "crop"],
    [max(x["vals_l2"]) for x in DATA if x["model"] == "crop"],
    label="Dim. Dropping", color="tab:red",
    linewidth=1.7,
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "crop"],
    [min(x["vals_l2"]) for x in DATA if x["model"] == "crop"],
    alpha=0.4, color="tab:red", linestyle="-",
    linewidth=1.7,
)

ax.plot(
    [768 - x["del_dim"] for x in DATA_GR],
    [x["val_l2"] for x in DATA_GR],
    label="Greedy Dim. Dropping", color="tab:red", linestyle=":",
    linewidth=1.7,
)

ax.plot(
    [x["dim"] for x in DATA if x["model"] == "sparse"],
    [max(x["vals_l2"]) for x in DATA if x["model"] == "sparse"],
    label="Sparse", color="tab:blue", linestyle="-",
    linewidth=1.7,
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "sparse"],
    [min(x["vals_l2"]) for x in DATA if x["model"] == "sparse"],
    alpha=0.4, color="tab:blue", linestyle="-",
    linewidth=1.7,
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "gauss"],
    [max(x["vals_l2"]) for x in DATA if x["model"] == "gauss"],
    label="Gaussian", color="tab:blue", linestyle=":",
    linewidth=1.7,
)
ax.plot(
    [x["dim"] for x in DATA if x["model"] == "gauss"],
    [min(x["vals_l2"]) for x in DATA if x["model"] == "gauss"],
    alpha=0.4, color="tab:blue", linestyle=":",
    linewidth=1.7,
)

# uncompressed
ax.axhline(y=0.431, alpha=0.5, linestyle="--", color="black")

# plt.legend(bbox_to_anchor=(-0.1, 1, 1.2, 0), loc="center", ncol=2)
plt.legend(ncol=2)

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimensions")

plt.tight_layout()
plt.show()

loss_against_avg_l2 = np.average(
    [max(x["vals_l2"]) - np.average(x["vals_l2"]) for x in DATA])
loss_against_avg_ip = np.average(
    [max(x["vals_ip"]) - np.average(x["vals_ip"]) for x in DATA])
print((loss_against_avg_ip + loss_against_avg_l2) / 2)
