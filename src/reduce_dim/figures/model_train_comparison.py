#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pca-logfile', default="computed/pca_time.log")
args = parser.parse_args()

with open(args.pca_logfile, "r") as f:
    DATA_PCA = eval(f.read())
SIZES = sorted(list(set([x["dim"] for x in DATA_PCA])), key=lambda x: int(x))
# DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.8, 3.2))
ax = plt.gca() 

ax.plot(
    SIZES, [x["val_ip"] for x in DATA_PCA],
    label="PCA R-Prec", color="tab:blue", linestyle="-"
)

# uncompressed
# ax.axhline(y=0.454, alpha=0.5, linestyle="--", color="black")

# plt.legend(bbox_to_anchor=(-0.1, 1, 1.2, 0), loc="center", ncol=2)
plt.legend(ncol=2)

# ax.set_xticks(DISPLAY_DIMS)
# ax.set_xticklabels(DISPLAY_DIMS)
# ax.set_ylabel("R-Precision")
# ax.set_xlabel("Dimension")

plt.tight_layout()
plt.show()


loss_against_avg_l2 = np.average([max(x["vals_l2"]) - np.average(x["vals_l2"]) for x in DATA_PCA])
loss_against_avg_ip = np.average([max(x["vals_ip"]) - np.average(x["vals_ip"]) for x in DATA_PCA])
print((loss_against_avg_ip+loss_against_avg_l2)/2)
