#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile-pca-scikit', default="computed/pca_speed_puffyemery.log")
parser.add_argument('--logfile-pca-torch', default="computed/pca_speed_puffyemery_torch.log")
parser.add_argument('--logfile-pca-torch-gpu', default="computed/pca_speed_puffyemery_torch_gpu.log")
parser.add_argument('--logfile-auto-gpu', default="computed/auto_speed_puffyemery_gpu.log")
parser.add_argument('--logfile-auto', default="computed/auto_speed_puffyemery.log")
parser.add_argument('--key', default=0, type=int)
args = parser.parse_args()

with open(args.logfile_pca_scikit, "r") as f:
    data_pca_scikit = eval(f.read())
with open(args.logfile_pca_torch, "r") as f:
    data_pca_torch = eval(f.read())
with open(args.logfile_pca_torch_gpu, "r") as f:
    data_pca_torch_gpu = eval(f.read())
with open(args.logfile_auto_gpu, "r") as f:
    data_auto_gpu = eval(f.read())
with open(args.logfile_auto, "r") as f:
    data_auto = eval(f.read())

DIMS = sorted(list(set([x["dim"] for x in data_pca_scikit])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.6, 4.7))
ax = plt.gca()

plt.rcParams["lines.linewidth"] = 2.2

ax.plot(DIMS, [x["encode_time"] for x in data_pca_scikit], label="PCA (scikit), encode", color="tab:green", linestyle="-", alpha=0.75)
ax.plot(DIMS, [x["encode_time"] for x in data_pca_torch_gpu], label="PCA (Torch, GPU), encode", color="tab:blue", linestyle="-", alpha=0.75)
ax.plot(DIMS, [x["encode_time"] for x in data_pca_torch], label="PCA (Torch, CPU), encode", color="tab:cyan", linestyle="-", alpha=0.75)
ax.plot(DIMS, [x["encode_time"] for x in data_auto_gpu], label="Auto. (GPU), encode", color="tab:red", linestyle="-", alpha=0.75)
ax.plot(DIMS, [x["encode_time"] for x in data_auto], label="Auto. (CPU), encode", color="tab:pink", linestyle="-", alpha=0.75)


ax.plot(DIMS, [x["train_time"] for x in data_pca_scikit], label="PCA (scikit), train", color="tab:green", linestyle=":")
ax.plot(DIMS, [x["train_time"] for x in data_pca_torch_gpu], label="PCA (Torch, GPU), train", color="tab:blue", linestyle=":")
ax.plot(DIMS, [x["train_time"] for x in data_pca_torch], label="PCA (Torch, CPU), train", color="tab:cyan", linestyle=":")
ax.plot(DIMS, [x["train_time"] for x in data_auto_gpu], label="Auto. (GPU), train", color="tab:red", linestyle=":")
ax.plot(DIMS, [x["train_time"] for x in data_auto], label="Auto. (CPU), train", color="tab:pink", linestyle=":")


ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("Time (s)")
ax.set_xlabel("Dimension")
# ax.set_ylim(0.015, 0.66)

h1, l1 = ax.get_legend_handles_labels()

# plt.title(["No pre-processing", "Normalized", "Centered", "Centered, Normalized"][args.key])
plt.legend(
    h1, l1,
    loc="center",
    bbox_to_anchor=(-0.05, 1.12, 1, 0.2),
    ncol=2,
    columnspacing=1.4,
)
plt.tight_layout(rect=(0, 0, 1, 1.05))
plt.savefig("figures/model_speed.pdf")
plt.show()