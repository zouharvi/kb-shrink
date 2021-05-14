#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())
DIMS = sorted(list(set([x["dim"] for x in DATA])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.8, 3.7))
ax = plt.gca() 
ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_doc"], label="IP, Docs", color="tab:blue")
ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_query"], label="IP, Queries", color="tab:red")
ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_both"], label="IP, Both", color="tab:purple")

ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_doc"], label="L2, Docs", color="tab:blue", linestyle=":")
ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_query"], label="L2, Queries", color="tab:red", linestyle=":")
ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_both"], label="L2, Both", color="tab:purple", linestyle=":")

plt.legend(bbox_to_anchor=(-0.1, 1, 1.2, 0), loc="lower left", mode="expand", ncol=3)

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimension")

ax2 = ax.twinx()
ax2.plot(DIMS, [(x["loss_d"]+x["loss_q"])/2 for x in DATA if x["type"] == "train_doc"], color="tab:blue", alpha=0.2)
ax2.plot(DIMS, [(x["loss_d"]+x["loss_q"])/2 for x in DATA if x["type"] == "train_query"], color="tab:red", alpha=0.2)
ax2.plot(DIMS, [(x["loss_d"]+x["loss_q"])/2 for x in DATA if x["type"] == "train_both"], color="tab:purple", alpha=0.2)
ax2.set_ylabel("Reconstruction loss")

plt.tight_layout()
plt.show()


# TODO: compute correlation of loss or plot it