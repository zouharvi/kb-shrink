#!/usr/bin/env python3

import sys
sys.path.append("src")
# import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument(
    "--logfile", default="computed/autofilter_classifier_clearwindow.json"
)
args = args.parse_args()

with open(args.logfile, "r") as f:
    data = json.load(f)

xpoints = list(range(1, len(data) + 1))
plt.figure(figsize=(5, 4))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(
    xpoints,
    [x["mccc"] for x in data],
    label="MCCC",
    color="gray",
    marker="^",
)

ax1.plot(
    xpoints,
    [x["mccc_total"] for x in data],
    label="MCCC (total)",
    color="silver",
    marker="x",
)

ax1.plot(
    xpoints,
    [x["lr"] for x in data],
    label="LR",
    color="tab:blue",
    marker="^",
)
ax1.plot(
    xpoints,
    [x["lr_total"] for x in data],
    label="LR (total)",
    color="lightsteelblue",
    marker="x",
)

ax2.scatter(
    xpoints,
    [x["positive"] for x in data],
    color="tab:green",
    label="Positive docs",
    s=15,
    edgecolors="black",
)
ax2.scatter(
    xpoints,
    [x["negative"] for x in data],
    color="tab:gray",
    label="Pruned docs",
    s=15,
    edgecolors="black",
)

ax1.set_ylabel("Classifier acc (cutoff axis)")
ax1.set_xlabel("Step (top-10)")
ax2.set_ylabel("Number of docs")
ax2.set_ylim(10e3, 350e3)
# ax1.set_ylim(0.5, 0.88)

ax1.set_xticks(
    xpoints,
    xpoints,
)

leg1h, leg1l = ax1.get_legend_handles_labels()
leg2h, leg2l = ax2.get_legend_handles_labels()
plt.legend(
    leg1h + leg2h,
    leg1l + leg2l,
    ncol=3,
    bbox_to_anchor=(-0.1, 1.02, 1, 0),
    loc="lower left",
    columnspacing=0.5,
)
plt.tight_layout()
plt.savefig(f"figures/autofilter_classifier.pdf")
plt.show()
