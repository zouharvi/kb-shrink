#!/usr/bin/env python3

"""
./src/reduce_dim/figures/speed.py --vert \
    --logfile-pca-scikit computed/speed/speed_pca_scikit_reddrapes_* \
    --logfile-pca-torch computed/speed/speed_pca_torch_cpu_reddrapes_* \
    --logfile-pca-torch-gpu computed/speed/speed_pca_torch_gpu_reddrapes_* \
    --logfile-auto-gpu computed/speed/speed_auto_torch_gpu_reddrapes_* \
    --logfile-auto computed/speed/speed_auto_torch_cpu_reddrapes_* 
"""

import sys
sys.path.append("src")
import misc.plot_utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.stats as st
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile-pca-scikit', default=["computed/pca_speed_puffyemery.log"], nargs="+")
parser.add_argument('--logfile-pca-torch', default=["computed/pca_speed_puffyemery_torch.log"], nargs="+")
parser.add_argument('--logfile-pca-torch-gpu', default=["computed/pca_speed_puffyemery_torch_gpu.log"], nargs="+")
parser.add_argument('--logfile-auto-gpu', default=["computed/auto_speed_puffyemery_gpu.log"], nargs="+")
parser.add_argument('--logfile-auto', default=["computed/auto_speed_puffyemery.log"], nargs="+")
parser.add_argument('--key', default=0, type=int)
parser.add_argument('--vert', action="store_true")
args = parser.parse_args()

def deep_merge(data_orig, data_new):
    """
    inefficient deep merge but works
    """
    if data_orig is None:
        data_orig = [{k: [v] for k, v in x.items()}
                        for x in data_new]
    else:
        # a complicated deep merge
        for x in data_new:
            data_orig = [
                # deep merge
                {
                    k: v + [x[k]]
                    for k, v
                    in xo.items()
                }
                # find the correct line
                if x["type"] == xo["type"][0] and x["dim"] == xo["dim"][0]
                else xo
                for xo in data_orig
            ]
    return data_orig

data_pca_scikit = None
data_pca_torch = None
data_pca_torch_gpu = None
data_auto_gpu = None
data_auto = None
for file in args.logfile_pca_scikit:
    with open(file, "r") as f:
        data_pca_scikit = deep_merge(data_pca_scikit, eval(f.read()))
for file in args.logfile_pca_torch:
    with open(file, "r") as f:
        data_pca_torch = deep_merge(data_pca_torch, eval(f.read()))
for file in args.logfile_pca_torch_gpu:
    with open(file, "r") as f:
        data_pca_torch_gpu = deep_merge(data_pca_torch_gpu, eval(f.read()))
for file in args.logfile_auto_gpu:
    with open(file, "r") as f:
        data_auto_gpu = deep_merge(data_auto_gpu, eval(f.read()))
for file in args.logfile_auto:
    with open(file, "r") as f:
        data_auto = deep_merge(data_auto, eval(f.read()))

DIMS = sorted(list(set([x["dim"][0] for x in data_pca_scikit])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4,4) if args.vert else (7, 5))
ax1 = plt.gca()
extra = Rectangle(
    (0, 0), 1, 1, fc="w",
    fill=False, edgecolor='none', linewidth=0
)

# plt.rcParams["lines.linewidth"] = 2.2

def plot_line_with_errorbars(key, data, label, color, ax, linestyle="-"):
    conf_intervals = [
        st.t.interval(
            alpha=0.95, df=len(x[key]) - 1,
            loc=np.average(x[key]), scale=st.sem(x[key])
        )
        for x in data
    ]
    conf_intervals = [(x[1] - x[0]) / 2 for x in conf_intervals]
    ax.errorbar(
        DIMS,
        [np.average(x[key])
        for x in data],
        yerr=conf_intervals,
        fmt="none",
        ecolor=color, elinewidth=1.5,
        alpha=0.2,
        capsize=4
    )
    ax.plot(DIMS, [np.average(x[key]) for x in data], label=label, color=color, linestyle=linestyle, alpha=0.75)

plot_line_with_errorbars("encode_time", data_pca_scikit, label="PCA (scikit)", color="tab:green", ax=ax1)
plot_line_with_errorbars("encode_time", data_pca_torch_gpu, label="PCA (Torch, GPU)", color="tab:blue", ax=ax1)
plot_line_with_errorbars("encode_time", data_pca_torch, label="PCA (Torch, CPU)", color="tab:cyan", ax=ax1)
plot_line_with_errorbars("encode_time", data_auto_gpu, label="Auto. (GPU)", color="tab:red", ax=ax1)
plot_line_with_errorbars("encode_time", data_auto, label="Auto. (CPU)", color="tab:pink", ax=ax1)
ax1.set_ylabel("Encode time (s)")
ax1.set_xlabel("Dimension")
ax1.set_xticks(DISPLAY_DIMS)
ax1.set_xticklabels(DISPLAY_DIMS)
if not args.vert:
    ax1.set_title("Encode Time")

if args.vert:
    plt.tight_layout(h_pad=0, rect=(0, 0, 1, 0.85))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.3),
        ncol=2,
        columnspacing=1.4,
    )
else:
    plt.tight_layout(h_pad=0, rect=(0,0,0.76,1))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        columnspacing=1.4,
    )

plt.savefig("figures/model_speed_encode.pdf")
plt.show()

plt.figure(figsize=(4,4) if args.vert else (7, 5))
ax2 = plt.gca()

plot_line_with_errorbars("train_time", data_pca_scikit, label="PCA (scikit)", color="tab:green", ax=ax2, linestyle="--")
plot_line_with_errorbars("train_time", data_pca_torch_gpu, label="PCA (Torch, GPU)", color="tab:blue", ax=ax2, linestyle="--")
plot_line_with_errorbars("train_time", data_pca_torch, label="PCA (Torch, CPU)", color="tab:cyan", ax=ax2, linestyle="--")
plot_line_with_errorbars("train_time", data_auto_gpu, label="Auto. (GPU)", color="tab:red", ax=ax2, linestyle="--")
plot_line_with_errorbars("train_time", data_auto, label="Auto. (CPU)", color="tab:pink", ax=ax2, linestyle="--")

ax2.set_ylabel("Train time (s)")
ax2.set_xlabel("Dimension")
ax2.set_xticks(DISPLAY_DIMS)
ax2.set_xticklabels(DISPLAY_DIMS)
if not args.vert:
    ax2.set_title("Train Time")

h2, l2 = ax2.get_legend_handles_labels()

if args.vert:
    plt.tight_layout(h_pad=0, rect=(0, 0, 1, 0.85))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.3),
        ncol=2,
        columnspacing=1.4,
    )
else:
    plt.tight_layout(h_pad=0, rect=(0,0,0.76,1))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        columnspacing=1.4,
    )
# ax.set_ylim(0.015, 0.66)

plt.savefig("figures/model_speed_train.pdf")
plt.show()