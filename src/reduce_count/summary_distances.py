#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import l2_sim, read_keys_pickle, vec_sim
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

similarities_ip = [
    sorted(sims, reverse=True)
    for sims in vec_sim(data, sim_func=np.inner)
]
similarities_ip = np.average(similarities_ip, axis=0)
similarities_l2 = [
    sorted(sims, reverse=True)
    for sims in vec_sim(data, sim_func=l2_sim)
]
similarities_l2 = np.average(similarities_l2, axis=0)

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny().twinx()
ax3 = ax1.twinx()

line2, = ax2.plot(
    similarities_ip[1:100],
    color="tab:orange",
)
line1, = ax1.plot(
    similarities_ip[1:],
    color="tab:blue",
)
# mark top 100 on the global line
line1_mark, = ax1.plot(
    similarities_ip[1:100],
    color="tab:orange",
    linewidth=3,
    linestyle="dotted"
)

line3, = ax3.plot(
    similarities_l2[1:],
    color="tab:green",
)
ax1.set_xlabel("Neighbour")
ax1.set_ylabel("Inner Product All")
ax2.get_yaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.set_ylabel("- L2 All")

plt.xlabel("Neighbours")

plt.title(f"Average Simlarities for Neighbours ({args.keys_in})")
plt.tight_layout()
plt.legend(
    [line1, line2, line1_mark, line3],
    ["All neighbours (IP)", "Top 100 Neighbours (IP)",
     "Top 100 Neighbours (IP)", "All neighbours (-L2)"],
    loc="lower left")
plt.show()