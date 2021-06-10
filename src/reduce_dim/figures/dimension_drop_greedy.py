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
DISPLAY_DIMS = [0, 256, 512, 768]

plt.figure(figsize=(4.8, 4.0))
ax = plt.gca() 
ax.plot([x["val_ip"] for x in DATA if x["metric"] == "l2"][1:], label="IP, (IP order)", color="tab:blue", linestyle="-")
ax.plot([x["val_l2"] for x in DATA if x["metric"] == "l2"][1:], label="L2, (IP order)", color="tab:red", linestyle="-")
ax.plot([x["val_ip"] for x in DATA if x["metric"] == "ip"][1:], label="IP, (IP order)", color="tab:blue", linestyle=":")
ax.plot([x["val_l2"] for x in DATA if x["metric"] == "ip"][1:], label="L2, (IP order)", color="tab:red", linestyle=":")


ax.axvline(x=550, alpha=0.25, linestyle="--", color="tab:blue")
ax.axvline(x=496, alpha=0.25, linestyle="--", color="tab:blue")

# uncompressed
ax.axhline(y=0.3229, alpha=0.5, linestyle="--", color="black")
# ax.set_ylim(0.2, 0.5)

plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Top-k dimensions dropped")

plt.tight_layout()
plt.show()