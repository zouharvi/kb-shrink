#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--logfile-pca', default="computed/pca_irrelevant_uneasiness.log")
parser.add_argument(
    '--logfile-auto', default=["computed/auto_irrelevant_uneasiness.log"],
    nargs="+"
)
parser.add_argument(
    '--logfile-uncompressed',
    default="computed/uncompressed_irrelevant_oilynoodles.log"
)
parser.add_argument('--key', default=0, type=int)
parser.add_argument('--vert', action="store_true")
args = parser.parse_args()

with open(args.logfile_pca, "r") as f:
    data_pca = eval(f.read())

data_auto = None
for file in args.logfile_auto:
    with open(file, "r") as f:
        if data_auto is None:
            data_auto = [
                {k: [v] for k, v in x.items()}
                for x in eval(f.read())
            ]
        else:
            # a complicated deep merge
            for x in eval(f.read()):
                data_auto = [
                    # deep merge
                    {
                        k: v + [x[k]]
                        for k, v
                        in xo.items()
                    }
                    # find the correct line
                    if x["type"] == xo["type"][0] and x["num_samples"] == xo["num_samples"][0]
                    else xo
                    for xo in data_auto
                ]
with open(args.logfile_uncompressed, "r") as f:
    data_uncompressed = eval(f.read())

DIMS = [
    np.log10(x["num_samples"])
    for x in data_pca if x["type"] == "train_data"
]
DIMS_NEXT = [
    np.log10(x["num_samples"])
    for x in data_pca if x["type"] == "eval_data"
]
DISPLAY_DIMS = [128, 10**3, (10**4), 10**5, 10**6, 10**7, (10**7) * 3]
THRESHOLD = 2114151

plt.figure(figsize=(4, 4) if args.vert else (7, 4))
ax = plt.gca()

ax.plot(
    DIMS,
    [x["val_ip"] for x in data_pca if x["type"] == "train_data"],
    label="PCA (training docs)", color="tab:blue", linestyle="-")

conf_intervals = [
    st.t.interval(
        alpha=0.95, df=len(x["val_ip"]) - 1,
        loc=np.average(x["val_ip"]), scale=st.sem(x["val_ip"])
    )
    for x in data_auto
    if x["type"][0] == "train_data"
]
conf_intervals = [(x[1] - x[0]) / 2 for x in conf_intervals]
ax.errorbar(
    DIMS,
    [np.average(x["val_ip"])
     for x in data_auto if x["type"][0] == "train_data"],
    yerr=conf_intervals,
    fmt="none",
    ecolor="tab:grey", elinewidth=1.5,
    capsize=4
)
ax.plot(
    DIMS,
    [np.average(x["val_ip"])
     for x in data_auto if x["type"][0] == "train_data"],
    label="Auto. (training docs)", color="tab:red", linestyle="-"
)


ax.hlines(
    0.619,
    xmin=np.log10(128), xmax=np.log10(DISPLAY_DIMS[-1]),
    color="gray",
    label="Uncompressed"
)
ax.plot(
    DIMS_NEXT,
    [x["val_ip"] for x in data_pca if x["type"] == "eval_data"],
    label="PCA (eval docs)", color="tab:blue", linestyle=":")
ax.plot(
    DIMS_NEXT,
    [x["val_ip"][0] for x in data_auto if x["type"][0] == "eval_data"],
    label="Auto. (eval docs)", color="tab:red", linestyle=":")

ax.plot(
    DIMS_NEXT,
    [x["val_ip"] for x in data_uncompressed if x["type"] == "eval_data"],
    label="Uncomp. (eval docs)", color="gray", linestyle=":")

ax.set_xticks([np.log10(x) for x in DISPLAY_DIMS])
ax.set_xticklabels([
    x if x == 128 else '$10^{' + f'{np.log10(x):.1f}' + '}$'
    for x in DISPLAY_DIMS
])
ax.set_ylabel("R-Precision")
ax.set_xlabel("Docs count (log scale)")
ax.set_ylim(0.37, 0.635)
ax.set_xlim(2, 7.6)

plt.scatter(
    np.log10(THRESHOLD),
    [
        x["val_ip"]
        for x in data_uncompressed
        if x["type"] == "eval_data" and x["num_samples"] == THRESHOLD
    ],
    marker="x", color="black", zorder=10, s=17,
)
plt.scatter(
    np.log10(THRESHOLD),
    [
        x["val_ip"]
        for x in data_pca
        if x["type"] == "train_data" and x["num_samples"] == THRESHOLD
    ],
    marker="x", color="black", zorder=10, s=17,
)
plt.scatter(
    np.log10(THRESHOLD),
    [
        x["val_ip"][0]
        for x in data_auto
        if x["type"][0] == "train_data" and x["num_samples"][0] == THRESHOLD
    ],
    marker="x", color="black", zorder=10, s=17
)

h1, l1 = ax.get_legend_handles_labels()


if args.vert:
    plt.legend(
        h1, l1,
        loc="upper left",
        bbox_to_anchor=(-0.15, 1.3),
        columnspacing=1.3,
        ncol=2,
    )
    plt.tight_layout(rect=(0, 0, 1, 1.05))
else:
    plt.legend(
        h1, l1,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
    )
    plt.tight_layout()

plt.savefig("figures/model_data.pdf")
plt.show()
