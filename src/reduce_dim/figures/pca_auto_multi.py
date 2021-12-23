#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--legend', action="store_true")
parser.add_argument('--logfile', nargs="+")
args = parser.parse_args()

data_all = []
for logfile in args.logfile:
    with open(logfile, "r") as f:
        data = eval(f.read())
        # detect autoencoder
        if "docs_loss" in data[0]:
            data = [
                {
                    **item,
                    "loss_d": item["docs_loss"],
                    "loss_q": item["queries_loss"]
                }
                for item in data
            ]
            is_autoencoder = True
        else:
            is_autoencoder = False
        data_all.append(data)

# take the first logfile as the main one
DIMS = sorted(list(set([x["dim"] for x in data_all[0]])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

# fig, axes = plt.subplots(1, 4, sharey=True, figsize=(10,3))
fig = plt.figure(figsize=(10, 3))

gs = gridspec.GridSpec(
    1, 4, width_ratios=[1, 1, 1, 1],
    wspace=0.05, hspace=0.0,
    top=0.9 if not args.legend else 0.1, bottom=0.15 if not args.legend else -1,
    left=0.06, right=0.94
)

for (key, data) in zip(range(4), data_all):
    if key == 0:
        uncompressed_ip = 0.609
        uncompressed_l2 = 0.240
    elif key == 1:
        uncompressed_ip = 0.463
        uncompressed_l2 = None
    elif key == 2:
        uncompressed_ip = 0.630
        uncompressed_l2 = 0.353
    elif key == 3:
        uncompressed_ip = 0.618
        uncompressed_l2 = None

    ax_main = plt.subplot(gs[0, key])

    ax1 = ax_main
    legend_ip_doc = ax1.plot(
        DIMS,
        [x["val_ip"] for x in data if x["type"] == "d"], label="IP, Docs", color="tab:blue", linestyle="-")
    legend_l2_doc = ax1.plot(
        DIMS,
        [x["val_l2"] for x in data if x["type"] == "d"], label="L2, Docs", color="tab:red", linestyle="-")

    legend_ip_query = ax1.plot(
        DIMS, [x["val_ip"] for x in data if x["type"] == "q"], label="IP, Queries", color="tab:blue", linestyle=":")
    legend_l2_query = ax1.plot(
        DIMS,
        [x["val_l2"] for x in data if x["type"] == "q"], label="L2, Queries", color="tab:red", linestyle=":")

    legend_ip_both = ax1.plot(
        DIMS,
        [x["val_ip"] for x in data if x["type"] == "dq"], label="IP, Both", color="tab:blue", linestyle="-.")
    legend_l2_both = ax1.plot(
        DIMS,
        [x["val_l2"] for x in data if x["type"] == "dq"], label="L2, Both", color="tab:red", linestyle="-.")

    ax1.set_xticks(DISPLAY_DIMS)
    ax1.set_xticklabels(DISPLAY_DIMS)
    ax1.set_ylabel("R-Precision " + ("(Autoencoder)" if is_autoencoder else "(PCA)"))
    # ax1.set_xlabel("Dimension")
    ax1.set_ylim(0.015, 0.66)

    ax2 = ax1.twinx()
    legend_loss_doc = ax2.plot(
        DIMS,
        [(x["loss_d"] + x["loss_q"]) / 2 for x in data if x["type"] == "d"], label="Loss Docs", color="tab:blue", alpha=0.2)
    legend_loss_query = ax2.plot(
        DIMS,
        [(x["loss_d"] + x["loss_q"]) / 2 for x in data if x["type"] == "q"], label="Loss Queries", color="tab:red", alpha=0.2)
    legend_loss_both = ax2.plot(
        DIMS,
        [(x["loss_d"] + x["loss_q"]) / 2 for x in data if x["type"] == "dq"], label="Loss Both", color="tab:purple", alpha=0.2)
    ax2.set_ylim(-0.001, 0.065)
    ax2.set_ylabel("Reconstruction loss")

    # uncompressed
    if uncompressed_l2 is not None:
        ax1.hlines(
            y=uncompressed_ip, xmin=32, xmax=768,
            alpha=0.5, linestyle="--", color="tab:blue")
        ax1.hlines(
            y=uncompressed_l2, xmin=32, xmax=768,
            alpha=0.5, linestyle="--", color="tab:red")
    else:
        h3 = ax1.hlines(
            y=uncompressed_ip, xmin=32, xmax=768, alpha=0.5,
            linestyle="--", color="black", label="Original")

    ax1.get_yaxis().set_visible(key == 0)
    ax2.get_yaxis().set_visible(key == 3)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    if not is_autoencoder:
        ax_main.title.set_text(
            [
                "No pre-processing", "Normalized",
                "Centered", "Centered, Normalized"
            ][key]
        )

if args.legend:
    plt.legend(
        h1 + h2,
        l1 + l2,
        bbox_to_anchor=(1, 1.5),
        ncol=5,
    )
    # plt.tight_layout()
plt.show()
