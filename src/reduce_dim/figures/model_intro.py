#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt

# done on --pruned
DATA_RPREC_PRUNED = {
    "DPR\n(Avg)": {
        "ip_fast": 0.408,
        "ip": 0.435,
        "l2_fast": 0.335,
        "l2": 0.338,
    },
    "Sentence\nBERT\n(Avg)": {
        "ip_fast": 0.308,
        "ip": 0.326,
        "l2_fast": 0.367,
        "l2": 0.377,
    },
    "BERT\n(Avg)": {
        "ip_fast": 0.157,
        "ip": 0.182,
        "l2_fast": 0.252,
        "l2": 0.261,
    },
    "DPR\n[CLS]": {
        "ip_fast": 0.609,
        "ip": 0.632,
        "l2_fast": 0.240,
        "l2": 0.244,
    },
    "Sentence\nBERT\n[CLS]": {
        "ip_fast": 0.324,
        "ip": 0.336,
        "l2_fast": 0.305,
        "l2": 0.313,
    },
    "BERT\n[CLS]": {
        "ip_fast": 0.007,
        "ip": 0.0038,
        "l2_fast": 0.028,
        "l2": 0.029,
    },
}

DATA_RPREC_BIGOLD = {
    "DPR\n(Avg)": {
        "ip_fast": 0.179,
        "ip": 0.193,
        "l2_fast": 0.165,
        "l2": 0.168,
    },
    "Sentence\nBERT\n(Avg)": {
        "ip_fast": 0.118,
        "ip": 0.125,
        "l2_fast": 0.176,
        "l2": 0.179,
    },
    "BERT\n(Avg)": {
        "ip_fast": 0.044,
        "ip": 0.053,
        "l2_fast": 0.096,
        "l2": 0.100,
    },
    "DPR\n[CLS]": {
        "ip_fast": 0.427,
        "ip": 0.446,
        "l2_fast": 0.092,
        "l2": 0.095,
    },
    "Sentence\nBERT\n[CLS]": {
        "ip_fast": 0.141,
        "ip": 0.146,
        "l2_fast": 0.140,
        "l2": 0.140,
    },
    "BERT\n[CLS]": {
        "ip_fast": 0.000,
        "ip": 0.000,
        "l2_fast": 0.003,
        "l2": 0.003,
    },
}

DATA_RPREC = DATA_RPREC_PRUNED
POS = np.arange(len(DATA_RPREC))
BARHEIGHT = 0.2
YERRMOCK = np.random.rand(len(DATA_RPREC))/10000

plt.figure(figsize=(4.4, 4))
ax = plt.gca()
ax.bar(
    POS + 0.2 * 0,
    [x["ip"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="IP",
    color="tab:red", hatch="",
    edgecolor="black",
)
ax.bar(
    POS + 0.2 * 1,
    [x["ip_fast"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="IP fast",
    color="pink", hatch="...",
    edgecolor="black",
)
ax.bar(
    POS + 0.2 * 2,
    [x["l2"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="$L^2$",
    color="tab:blue", hatch="",
    edgecolor="black",
)
ax.bar(
    POS + 0.2 * 3,
    [x["l2_fast"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="$L^2$ fast",
    color="lightblue", hatch="...",
    edgecolor="black",
)
ax.set_xticks(POS + 0.15 * 2)
ax.set_xticklabels(DATA_RPREC.keys())
ax.set_ylabel("R-Precision")
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
plt.tight_layout()
plt.savefig("figures/model_intro.pdf")
plt.show()
