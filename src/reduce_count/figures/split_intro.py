#!/usr/bin/env python3

import sys
sys.path.append("src")
# import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ACC 10
DATA = {
    "fixed040_cn": {"pcount": 3814169, "val_ip": 0.69375},
    "fixed060_cn": {"pcount": 2858954, "val_ip": 0.7},
    "fixed080_cn": {"pcount": 2385695, "val_ip": 0.7032142857142857},
    "fixed080_over.20_cn": {"pcount": 2385695, "val_ip": 0.7033928571428572},
    "fixed100_cn": {"pcount": 2114151, "val_ip": 0.7066071428571429},
    "fixed120_cn": {"pcount": 1947229, "val_ip": 0.705},
    "sent01_cn": {"pcount": 5665645, "val_ip": 0.6816071428571429},
    "sent02_cn": {"pcount": 3311972, "val_ip": 0.7028571428571428},
    "sent03_cn": {"pcount": 2542124, "val_ip": 0.7057142857142857},
    "sent04_cn": {"pcount": 2171949, "val_ip": 0.7117857142857142},
    "sent05_cn": {"pcount": 1965217, "val_ip": 0.7082142857142857},
    "sent06_cn": {"pcount": 1842948, "val_ip": 0.70875},
}
DATA_ACC20 = {
    "fixed040_cn": {"pcount": 3814169, "val_ip": 0.7533928571428572},
    "fixed060_cn": {"pcount": 2858954, "val_ip": 0.7573214285714286},
    "fixed080_cn": {"pcount": 2385695, "val_ip": 0.7567857142857143},
    "fixed080_over.20_cn": {"pcount": 2385695, "val_ip": 0.75625},
    "fixed100_cn": {"pcount": 2114151, "val_ip": 0.7560714285714286},
    "fixed120_cn": {"pcount": 1947229, "val_ip": 0.75625},
    "sent01_cn": {"pcount": 5665645, "val_ip": 0.7432142857142857},
    "sent02_cn": {"pcount": 3311972, "val_ip": 0.7625},
    "sent03_cn": {"pcount": 2542124, "val_ip": 0.7605357142857143},
    "sent04_cn": {"pcount": 2171949, "val_ip": 0.7623214285714286},
    "sent05_cn": {"pcount": 1965217, "val_ip": 0.7580357142857143},
    "sent06_cn": {"pcount": 1842948, "val_ip": 0.7582142857142857},
}

plt.figure(figsize=(5, 4))
ax = plt.gca()

COLOR_FIXED = list(mcolors.TABLEAU_COLORS)
COLOR_SENT = list(mcolors.TABLEAU_COLORS)


def get_style(name):
    style = {}
    if "fixed" in name:
        style["marker"] = "."
        style["color"] = COLOR_FIXED.pop(0)
    elif "sent" in name:
        style["marker"] = "^"
        style["color"] = COLOR_SENT.pop(0)
    return style


def mangle_name(name):
    name = name.replace("_cn", "")
    name = name.replace("fixed", "Fixed ")
    name = name.replace("_over.", ", over. ")
    name = name.replace("sent", "Sent ")
    name = name.replace(" 0", " ")
    return name


for name, vals in DATA.items():
    ax.scatter(
        vals["pcount"], vals["val_ip"],
        label=mangle_name(name), **get_style(name)
    )

ax.set_ylabel("Acc-10")
ax.set_xlabel("Passage count")
xpoints = np.arange(2e6, 5e6 + 1, 1e6)

plt.xticks(xpoints, [f"{x/(1e6):.0f}m" for x in xpoints])
plt.legend(
    ncol=2,
    # loc="lower left",
    columnspacing=0.5, 
    handlelength=0.5,
    handletextpad=0.5,
)
plt.tight_layout()
plt.savefig("figures/split_intro.pdf")
plt.show()
