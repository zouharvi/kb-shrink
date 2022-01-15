#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/prec_pca_mult.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())

DATA = [x for x in DATA]


DIMS = sorted(list(set([x["dim"] for x in DATA])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.4, 3))
ax = plt.gca()

ax.scatter(
    [x["dim"] for x in DATA if x["bit"] == "1"],
    [x["val_ip"] for x in DATA if x["bit"] == "1"],
    label="1-bit", color="tab:red",
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
plt.savefig("figures/prec_pca_mult.pdf")
plt.show()