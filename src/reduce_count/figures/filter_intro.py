#!/usr/bin/env python3

import sys
sys.path.append("src")
# import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

DATA = {
    "word2": {"pcount": 2083586, "val_ip": 0.6910714285714286},
    "word10": {"pcount": 1863866, "val_ip": 0.5153571428571428},
    "word20": {"pcount": 1683733, "val_ip": 0.5026785714285714},
    "word50": {"pcount": 1141334, "val_ip": 0.4305357142857143},
    "word100": {"pcount": 473530, "val_ip": 0.2741071428571429},
    "No filter": {"pcount": 2114151, "val_ip": 0.7008928571428571},
    "char10": {"pcount": 2082723, "val_ip": 0.6930357142857143},
    "char50": {"pcount": 1885526, "val_ip": 0.5166071428571428},
    "char200": {"pcount": 1435703, "val_ip": 0.4780357142857143},
    "char400": {"pcount": 882702, "val_ip": 0.3775},
    "char500": {"pcount": 659231, "val_ip": 0.32821428571428574},
}

plt.figure(figsize=(5, 4))
ax = plt.gca()

COLOR_WORD = list(mcolors.TABLEAU_COLORS)
COLOR_CHAR = list(mcolors.TABLEAU_COLORS)


def get_style(name):
    style = {}
    if "word" in name:
        style["marker"] = "."
        style["color"] = COLOR_WORD.pop(0)
    elif "char" in name:
        style["marker"] = "^"
        style["color"] = COLOR_CHAR.pop(0)
    elif "No filter" == name:
        style["marker"] = "x"
        style["color"] = "tab:gray"
    return style


def mangle_name(name):
    name = name.replace("word", "Word >")
    name = name.replace("char", "Char >")
    return name


for name, vals in DATA.items():
    ax.scatter(
        vals["pcount"], vals["val_ip"],
        alpha=0.8,
        label=mangle_name(name), **get_style(name)
    )

ax.set_ylabel("Acc-10")
ax.set_xlabel("Passage count")
xpoints = np.arange(4e5, 25e5, 5e5)

plt.xticks(xpoints, [f"{x/(1e6):.1f}m" for x in xpoints])
plt.legend(
    ncol=2,
    # loc="lower left",
    columnspacing=0.5, 
    handlelength=0.5,
    handletextpad=0.5,
)
plt.tight_layout()
plt.savefig("figures/filter_intro.pdf")
plt.show()
