#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pca-logfile', default="computed/pca_time.log")
parser.add_argument('--asingle-logfile', default="computed/asingle_time.log")
parser.add_argument('--ashallow-logfile', default="computed/ashallow_time.log")
args = parser.parse_args()

with open(args.pca_logfile, "r") as f:
    DATA_PCA = eval(f.read())
SIZES_PCA = [x["threshold"] for x in DATA_PCA]

with open(args.asingle_logfile, "r") as f:
    DATA_ASINGLE = eval(f.read())
SIZES_ASINGLE = [x["threshold"] for x in DATA_ASINGLE]

# with open(args.ashallow_logfile, "r") as f:
#     DATA_ASHALLOW = eval(f.read())
# SIZES_ASHALLOW = [x["threshold"] for x in DATA_ASHALLOW]

DISPLAY_DIMS = [1000, 10000, 30000, 50000, 70000, 90000, 110000]

plt.figure(figsize=(4.8, 3.2))
ax1 = plt.gca()
ax2 = ax1.twinx()

handle1p, = ax1.plot(
    SIZES_PCA, [x["val_ip"] for x in DATA_PCA],
    label="PCA R-Prec", color="tab:blue", linestyle="-"
)
handle1t, = ax2.plot(
    SIZES_PCA, [x["train_time"] for x in DATA_PCA],
    label="PCA time", color="tab:blue", linestyle=":"
)

handle2p, = ax1.plot(
    SIZES_ASINGLE, [x["val_ip"] for x in DATA_ASINGLE],
    label="Autoencoder (single) R-Prec", color="tab:red", linestyle="-"
)
handle2t, = ax2.plot(
    SIZES_ASINGLE, [x["train_time"] for x in DATA_ASINGLE],
    label="Autoencoder (single) time", color="tab:red", linestyle=":"
)

# handle2p, = ax1.plot(
#     SIZES_ASHALLOW, [x["val_ip"] for x in DATA_ASHALLOW],
#     label="Autoencoder (shallow) R-Prec", color="tab:green", linestyle="-"
# )
# handle2t, = ax2.plot(
#     SIZES_ASHALLOW, [x["train_time"] for x in DATA_ASHALLOW],
#     label="Autoencoder (single) time", color="tab:green", linestyle=":"
# )

# uncompressed
# ax.axhline(y=0.454, alpha=0.5, linestyle="--", color="black")

# plt.legend(bbox_to_anchor=(-0.1, 1, 1.2, 0), loc="center", ncol=2)
handleAll = [handle1p, handle1t, handle2p, handle2t]
plt.legend(
    handleAll, [x.get_label() for x in handleAll], ncol=2,
    bbox_to_anchor=(-0.1, 1.2, 1.2, -0.1), loc="center"
)

ax1.set_xticks(DISPLAY_DIMS)
ax1.set_xticklabels([f"{x//1000}k" for x in DISPLAY_DIMS])
# ax.set_ylabel("R-Precision")
# ax.set_xlabel("Dimension")

plt.tight_layout(rect=(-0.03, -0.04, 1.02, 1.03))
plt.show()