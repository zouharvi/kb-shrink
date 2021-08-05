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
DISPLAY_DIMS = [32, 256, 512, 768]
DATA = [x for x in DATA if x["del_dim"] != False]

plt.figure(figsize=(4.8, 3.2))
ax = plt.gca() 
ax.plot(
    [x["del_dim"] for x in DATA if x["metric"] == "l2"],
    [x["val_ip"] for x in DATA if x["metric"] == "l2"][::-1],
    label="IP, (L2 order)", color="tab:blue", linestyle="-", alpha=0.5)
ax.plot(
    [x["del_dim"] for x in DATA if x["metric"] == "l2"],
    [x["val_l2"] for x in DATA if x["metric"] == "l2"][::-1],
    label="L2, (L2 order)", color="tab:red", linestyle="-", alpha=0.5)
ax.plot(
    [x["del_dim"] for x in DATA if x["metric"] == "ip"],
    [x["val_ip"] for x in DATA if x["metric"] == "ip"][::-1],
    label="IP, (IP order)", color="tab:blue", linestyle=":")
ax.plot(
    [x["del_dim"] for x in DATA if x["metric"] == "ip"],
    [x["val_l2"] for x in DATA if x["metric"] == "ip"][::-1],
    label="L2, (IP order)", color="tab:red", linestyle=":")


# ax.axvline(x=550, alpha=0.25, linestyle="--", color="tab:blue")
# ax.axvline(x=496, alpha=0.25, linestyle="--", color="tab:blue")

# uncompressed
ax.axhline(y=0.454, alpha=0.5, linestyle="--", color="black")

plt.legend(ncol=2)

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimensions preserved")
ax.set_ylim(-0.005, 0.465)

plt.tight_layout()
plt.show()