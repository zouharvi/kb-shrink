#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--uncompressed', type=float, default=0.635)
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())
DIMS = sorted(list(set([x["dim"] for x in DATA])), key=lambda x: int(x))
DISPLAY_DIMS = [32, 256, 512, 768]

plt.figure(figsize=(4.6, 3.5))
ax = plt.gca() 
legend_ip_doc = ax.plot(DIMS, [x["val_l2"] for x in DATA], label="Docs", color="tab:blue", linestyle="-")
plt.gcf().subplots_adjust(left=0.14, right=0.86, top=0.93, bottom=0.13)

# uncompressed
ax.axhline(y=args.uncompressed, alpha=0.5, linestyle="--", color="black")

ax.set_xticks(DISPLAY_DIMS)
ax.set_xticklabels(DISPLAY_DIMS)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Dimension")
# ax.set_ylim(0.015, 0.46)

ax2 = ax.twinx()
legend_loss_doc = ax2.plot(DIMS, [(x["queries_loss"]+x["docs_loss"])/2 for x in DATA ], label="Loss Docs", color="tab:blue", alpha=0.2)
# ax2.set_ylim(-0.001, 0.065)
ax2.set_ylabel("Reconstruction loss")


h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.title("Autoencoder (smaller data!)")
# plt.legend(
#     h1+h2,
#     l1+l2,
#     bbox_to_anchor=(1.5, 0.5, 0.25, 0.25),
#     ncol=1
# )
plt.tight_layout()
plt.show()