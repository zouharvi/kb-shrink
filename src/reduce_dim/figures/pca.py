#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--uncompressed-ip', type=float, default=0.3229)
parser.add_argument('--uncompressed-l2', type=float, default=None)
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())
DIMS = sorted(list(set([x["dim"] for x in DATA])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.6, 3.5))
ax = plt.gca() 
legend_ip_doc = ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_doc"], label="IP, Docs", color="tab:blue", linestyle="-")
legend_l2_doc = ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_doc"], label="L2, Docs", color="tab:red", linestyle="-")

legend_ip_query = ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_query"], label="IP, Queries", color="tab:blue", linestyle=":")
legend_l2_query = ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_query"], label="L2, Queries", color="tab:red", linestyle=":")

legend_ip_both = ax.plot(DIMS, [x["val_ip"] for x in DATA if x["type"] == "train_both"], label="IP, Both", color="tab:blue", linestyle="-.")
legend_l2_both = ax.plot(DIMS, [x["val_l2"] for x in DATA if x["type"] == "train_both"], label="L2, Both", color="tab:red", linestyle="-.")
plt.gcf().subplots_adjust(left=0.14, right=0.86, top=0.93, bottom=0.13)

# uncompressed
if args.uncompressed_l2 is not None:
    ax.axhline(y=args.uncompressed_ip, alpha=0.5, linestyle="--", color="tab:blue")
    ax.axhline(y=args.uncompressed_l2, alpha=0.5, linestyle="--", color="tab:red")
else:
    ax.axhline(y=args.uncompressed_ip, alpha=0.5, linestyle="--", color="black")


ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimension")
ax.set_ylim(0.015, 0.46)

ax2 = ax.twinx()
legend_loss_doc = ax2.plot(DIMS, [(x["loss_d"]+x["loss_q"])/2 for x in DATA if x["type"] == "train_doc"], label="Loss Docs", color="tab:blue", alpha=0.2)
legend_loss_query = ax2.plot(DIMS, [(x["loss_d"]+x["loss_q"])/2 for x in DATA if x["type"] == "train_query"], label="Loss Queries", color="tab:red", alpha=0.2)
legend_loss_both = ax2.plot(DIMS, [(x["loss_d"]+x["loss_q"])/2 for x in DATA if x["type"] == "train_both"], label="Loss Both", color="tab:purple", alpha=0.2)
ax2.set_ylim(-0.001, 0.065)
ax2.set_ylabel("Reconstruction loss")


h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.title("Centered")
# plt.legend(
#     h1+h2,
#     l1+l2,
#     bbox_to_anchor=(1.5, 0.5, 0.25, 0.25),
#     ncol=1
# )
# plt.tight_layout()
plt.show()