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
# DISPLAY_DIMS = [128, 256, 256+128, 512, 512+128, 768]

plt.figure(figsize=(4.8, 3.2))
ax = plt.gca() 
ax.plot([x["val_ip"] for x in DATA if x["metric"] == "l2" if x["del_dim"] <= 768][1:], label="IP, (L2 order)", color="tab:blue", linestyle="-", alpha=0.5)
ax.plot([x["val_l2"] for x in DATA if x["metric"] == "l2" if x["del_dim"] <= 768][1:], label="L2, (L2 order)", color="tab:red", linestyle="-", alpha=0.5)
ax.plot([x["val_ip"] for x in DATA if x["metric"] == "ip" if x["del_dim"] <= 768][1:], label="IP, (IP order)", color="tab:blue", linestyle=":")
ax.plot([x["val_l2"] for x in DATA if x["metric"] == "ip" if x["del_dim"] <= 768][1:], label="L2, (IP order)", color="tab:red", linestyle=":")


# ax.axvline(x=550, alpha=0.25, linestyle="--", color="tab:blue")
# ax.axvline(x=496, alpha=0.25, linestyle="--", color="tab:blue")

# uncompressed
ax.axhline(y=0.4533, alpha=0.5, linestyle="--", color="black")

plt.legend(loc="lower left", ncol=2)

ax.set_xticks(DISPLAY_DIMS[::-1])
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimensions preserved")
ax.set_ylim(-0.005, 0.465)

plt.tight_layout()
plt.show()