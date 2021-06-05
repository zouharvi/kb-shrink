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

plt.figure(figsize=(4.8, 3.2))
ax = plt.gca() 

ax.plot(
    DIMS, [max(x["vals_ip"]) for x in DATA if x["model"] == "crop"],
    label="IP, Crop", color="tab:blue", linestyle="-"
)
ax.plot(
    DIMS, [max(x["vals_ip"]) for x in DATA if x["model"] == "sparse"],
    label="IP, Sparse", color="tab:blue", linestyle="-."
)
ax.plot(
    DIMS, [max(x["vals_ip"]) for x in DATA if x["model"] == "gauss"],
    label="IP, Gaussian", color="tab:blue", linestyle=":"
)
ax.plot(
    DIMS, [max(x["vals_l2"]) for x in DATA if x["model"] == "crop"],
    label="L2, Crop", color="tab:red", linestyle="-"
)
ax.plot(
    DIMS, [max(x["vals_l2"]) for x in DATA if x["model"] == "sparse"],
    label="L2, Sparse", color="tab:red", linestyle="-."
)
ax.plot(
    DIMS, [max(x["vals_l2"]) for x in DATA if x["model"] == "gauss"],
    label="L2, Gaussian", color="tab:red", linestyle=":"
)

# min
# ax.scatter(
#     DIMS, [min(x["vals_ip"]) for x in DATA if x["model"] == "sparse"],
#     marker="_", color="tab:blue"
# )
# ax.scatter(
#     DIMS, [min(x["vals_l2"]) for x in DATA if x["model"] == "sparse"],
#     marker="_", color="tab:red"
# )

# ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_query"], label="IP, Queries", color="tab:blue", linestyle=":")
# ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_query"], label="L2, Queries", color="tab:red", linestyle=":")

# ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_both"], label="IP, Both", color="tab:blue", linestyle="-.")
# ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_both"], label="L2, Both", color="tab:red", linestyle="-.")

# uncompressed
ax.axhline(y=0.3229, alpha=0.5, linestyle="--", color="black")
# ax.axhline(y=0.2610, alpha=0.5, linestyle="--", color="tab:red")

# plt.legend(bbox_to_anchor=(-0.1, 1, 1.2, 0), loc="center", ncol=2)
plt.legend(ncol=2)

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimension")

plt.tight_layout()
plt.show()


loss_against_avg_l2 = np.average([max(x["vals_l2"]) - np.average(x["vals_l2"]) for x in DATA])
loss_against_avg_ip = np.average([max(x["vals_ip"]) - np.average(x["vals_ip"]) for x in DATA])
print((loss_against_avg_ip+loss_against_avg_l2)/2)