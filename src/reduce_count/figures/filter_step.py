#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("--logfile", default="computed/autofilter_whitesocket.json")
args.add_argument("--title-step", default="10")
args.add_argument("--title-train", action="store_true")
args = args.parse_args()

with open(args.logfile, "r") as f:
    data = json.load(f)

xpoints = list(range(len(data)))
plt.figure(figsize=(5, 4))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.bar(
    xpoints,
    [x["acc"] for x in data],
    label="Accuracy",
    edgecolor="black",
)

ax2.scatter(
    xpoints[1:],
    [x["positive"] for x in data[1:]],
    color="tab:green",
    label="Positive docs",
    alpha=0.8,
    edgecolors="black",
)
ax2.scatter(
    xpoints[1:],
    [x["negative"] for x in data[1:]],
    color="tab:red",
    label="Negative docs",
    alpha=0.8,
    edgecolors="black",
)
ax2.scatter(
    xpoints[1:],
    [x["to_prune"] for x in data[1:]],
    color="tab:gray",
    label="Pruned docs",
    alpha=0.8,
    edgecolors="black",
)
# ax2.scatter(
#     xpoints[1:],
#     [x["docs_old"] for x in data[1:]],
#     color="tab:gray"
# )

ax1.set_ylim(0.5, 0.88)
ax1.set_ylabel("Acc-10 (cutoff axis)")
ax1.set_xlabel("Step (top-10)")
ax2.set_ylim(10e3, 355e3)
ax2.set_ylabel("Number of docs")

ax1.set_xticks(
    xpoints,
    xpoints,
)

leg1h, leg1l = ax1.get_legend_handles_labels()
leg2h, leg2l = ax2.get_legend_handles_labels()
plt.tight_layout(rect=(0,0,1,0.95))
plt.legend(
    leg1h + leg2h,
    leg1l + leg2l,
    ncol=4,
    bbox_to_anchor=(0.03, 1.01, 1, 0),
    loc="lower center",
    columnspacing=0.5,
)
plt.savefig(f"figures/autofilter_train_dev.pdf")
plt.show()
