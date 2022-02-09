#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("--logfile", default="computed/analyzespans_onemare.json")
args = args.parse_args()

with open(args.logfile, "r") as f:
    data = json.load(f)

data_w = [[int(k)]*v for k,v in data["words"].items()]
data_w = [x for l in data_w for x in l]
data_c = [[int(k)]*v for k,v in data["chars"].items()]
data_c = [x for l in data_c for x in l]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

xpoints_w = np.arange(0, 100+1, 5)

ax1.hist(
    data_w,
    bins=xpoints_w,
    label="Words",
    edgecolor="black",
)
ax1.set_xticks(xpoints_w[::2], xpoints_w[::2])
ax1.set_ylim(0, 550000)
ax1.set_xlabel("Words")
ax1.set_ylabel("Count")

xpoints_c = np.arange(0, 900+1, 50)

ax2.hist(
    data_c,
    bins=xpoints_c,
    label="Chars",
    edgecolor="black",
)
ax2.set_xticks(xpoints_c[::2], xpoints_c[::2])
ax2.set_ylim(0, 550000)
ax2.get_yaxis().set_visible(False)
ax2.set_xlabel("Chars")

plt.tight_layout()
plt.savefig(f"figures/filter_distribution.pdf")
plt.show()
