#!/usr/bin/env python3

from collections import defaultdict
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


DIMS = sorted(list(set([x["dim"] for x in DATA if x["dim"] != None])), key=lambda x: int(x))
DISPLAY_DIMS = [128, 256, 384, 512, 640, 768]

plt.figure(figsize=(7, 4))
ax = plt.gca()

HANDLES = {}
STYLE = {
    1: {"label": "1-bit", "color": "tab:red"},
    8: {"label": "8-bit", "color": "tab:purple"},
    16: {"label": "16-bit", "color": "tab:blue"},
    32: {"label": "32-bit", "color": "tab:gray"},
}

STYLE_DIM = defaultdict(dict)
STYLE_DIM[None] = {"marker": "x"}

DATA.reverse()

def markersize(comp):
    if comp > 150:
        return 100
    elif comp > 90:
        return 80
    elif comp > 50:
        return 55
    elif comp > 25:
        return 45
    elif comp > 20:
        return 35
    elif comp > 15:
        return 30
    elif comp > 10:
        return 25
    elif comp > 5:
        return 15
    return 10

for x in DATA:
    if x["dim"] is None:
        continue

    handle = ax.scatter(
        x["dim"], x["val_ip"],
        s=markersize(x["compression"]),
        alpha=0.5,
        **STYLE[x["bit"]],
    )
    HANDLES[x["bit"]] = handle

    # show labels only for some points
    if x["bit"] in [1, 8]:
        ax.text(
            x["dim"], x["val_ip"]-0.02-x["compression"]/20000, str(x["compression"]).rstrip(".0"),
            horizontalalignment='center',
        )

# uncompressed
handle_uncompressed_1 = ax.axhline(
    y=[x["val_ip"] for x in DATA if x["dim"] is None and x["bit"] == 1][0],
    alpha=0.5, linestyle="--", color="tab:red"
)
ax.text(
    799, 0.565, "32",
    horizontalalignment='right',
    color="tab:red",
)
handle_uncompressed_8 = ax.axhline(
    y=[x["val_ip"] for x in DATA if x["dim"] is None and x["bit"] == 8][0],
    alpha=0.5, linestyle="--", color="tab:purple"
)
ax.text(
    799, 0.62, "4",
    horizontalalignment='right',
    color="tab:blue",
)
handle_uncompressed_16 = ax.axhline(
    y=[x["val_ip"] for x in DATA if x["dim"] is None and x["bit"] == 16][0],
    alpha=0.5, linestyle="--", color="tab:blue"
)
handle_uncompressed_32 = ax.axhline(
    y=[x["val_ip"] for x in DATA if x["dim"] is None and x["bit"] == 32][0],
    alpha=0.5, linestyle="--", color="tab:gray"
)

plt.legend(
    [
        handle_uncompressed_1,
        handle_uncompressed_8,
        handle_uncompressed_16,
        handle_uncompressed_32,
        HANDLES[1],
        HANDLES[8],
        HANDLES[16],
        HANDLES[32],
    ],
    [
        "1-bit (no PCA)",
        "8-bit (no PCA)",
        "16-bit (no PCA)",
        "32-bit (no PCA)",
        "PCA + 1-bit",
        "PCA + 8-bit",
        "PCA + 16-bit",
        "PCA + 32-bit",
    ],
    ncol=1,
    loc="upper left",
    bbox_to_anchor=(1, 1),
)

plt.xlim(100)
plt.ylim(0.35, 0.64)
ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimensions")

plt.tight_layout()
plt.savefig("figures/prec_pca_mult.pdf")
plt.show()