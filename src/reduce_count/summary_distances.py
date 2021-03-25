#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import mrr, read_keys_pickle, vec_sim, vec_sim_order
import argparse
from scipy.spatial.distance import minkowski
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reduce KB key count')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
args = parser.parse_args()

data = read_keys_pickle(args.keys_in)

similarities_p = [
    sorted(sims, reverse=True)
    for sims in vec_sim(data, np.inner)
]
similarities_p = np.average(similarities_p, axis=0)
distances_m = [sorted(sims, reverse=True) for sims in vec_sim(data, minkowski)]
distances_m = np.average(distances_m, axis=0)
max_distance_m = max(distances_m)
similarities_m = [max_distance_m - x for x in distances_m]

order_p = vec_sim_order(data, np.inner)
# adding a constant (max) is irrelevant for ordering
order_m = vec_sim_order(data, lambda x, y: -minkowski(x, y))

mrr_val = mrr(order_p, order_m, 20, report=True)

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny().twinx()
ax3 = ax1.twinx()

line2, = ax2.plot(
    similarities_p[:100],
    color="tab:orange",
)
line1, = ax1.plot(
    similarities_p,
    color="tab:blue",
)
# mark top 100 on the global line
line1_mark, = ax1.plot(
    similarities_p[:100],
    color="tab:orange",
    linewidth=3,
    linestyle="dotted"
)

line3, = ax3.plot(
    similarities_m,
    color="tab:green",
)
ax1.set_ylabel("Inner Product All")
ax2.get_yaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.set_ylabel("max - L2 All")

plt.xlabel("Neighbours")

plt.title(f"Average Simlarities for Neighbours ({args.keys_in})")
plt.tight_layout()
plt.legend(
    [line1, line2, line1_mark, line3],
    ["All neighbours (IP)", "Top 100 Neighbours (IP)",
     "Top 100 Neighbours (IP)", "All neighbours (max - L2)"],
    loc="lower left")
plt.show()
