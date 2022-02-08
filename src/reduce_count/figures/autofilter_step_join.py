#!/usr/bin/env python3

"""
Intended primarily for thesis
"""

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("--logfile-1", default="computed/autofilter_olivewall.json")
args.add_argument("--logfile-2", default="computed/autofilter_closepumpkin.json")
args = args.parse_args()

with open(args.logfile_1, "r") as f:
    dataA = json.load(f)
with open(args.logfile_2, "r") as f:
    dataB = json.load(f)

xpoints = list(range(len(dataA)))
fig, (axA1, axB1) = plt.subplots(1, 2, figsize=(7, 3.5))
axA2 = axA1.twinx()

axA1.bar(
    xpoints,
    [x["acc"] for x in dataA],
    label="Accuracy",
    edgecolor="black",
)

axA2.scatter(
    xpoints[1:],
    [x["positive"] for x in dataA[1:]],
    color="tab:green",
    label="Positive docs",
    alpha=0.8,
    edgecolors="black",
)
axA2.scatter(
    xpoints[1:],
    [x["negative"] for x in dataA[1:]],
    color="tab:red",
    label="Negative docs",
    alpha=0.8,
    edgecolors="black",
)
axA2.scatter(
    xpoints[1:],
    [x["to_prune"] for x in dataA[1:]],
    color="tab:gray",
    label="Pruned docs",
    alpha=0.8,
    edgecolors="black",
)

axA1.set_ylim(0.5, 0.88)
axA1.set_ylabel("Acc-10 (cutoff axis)")
axA1.set_xlabel("Step (top-10)")
axA2.set_ylim(8e3, 9*10e3)
axA2.get_yaxis().set_visible(False)

axA1.set_xticks(
    xpoints,
    xpoints,
)


axB2 = axB1.twinx()

axB1.bar(
    xpoints,
    [x["acc"] for x in dataB],
    label="Accuracy",
    edgecolor="black",
)

axB2.scatter(
    xpoints[1:],
    [x["positive"] for x in dataB[1:]],
    color="tab:green",
    label="Positive docs",
    alpha=0.8,
    edgecolors="black",
)
axB2.scatter(
    xpoints[1:],
    [x["negative"] for x in dataB[1:]],
    color="tab:red",
    label="Negative docs",
    alpha=0.8,
    edgecolors="black",
)
axB2.scatter(
    xpoints[1:],
    [x["to_prune"] for x in dataB[1:]],
    color="tab:gray",
    label="Pruned docs",
    alpha=0.8,
    edgecolors="black",
)

axB1.set_ylim(0.5, 0.88)
axB1.get_yaxis().set_visible(False)
axB1.set_xlabel("Step (top-20)")
axB2.set_ylim(8e3, 9*10e3)
axB2.set_ylabel("Number of docs")

axB1.set_xticks(
    xpoints,
    xpoints,
)

leg1h, leg1l = axA1.get_legend_handles_labels()
leg2h, leg2l = axA2.get_legend_handles_labels()
plt.tight_layout(rect=(0,0,1,0.95))
plt.legend(
    leg1h + leg2h,
    leg1l + leg2l,
    ncol=4,
    bbox_to_anchor=(-0.95, 1, 1, 0),
    loc="lower left",
    columnspacing=0.5,
)
plt.savefig(f"figures/autofilter_dev_dev.pdf")
plt.show()
