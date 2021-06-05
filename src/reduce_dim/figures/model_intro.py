#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt

DATA_RPREC = {
    "DPR\n(Avg)": {
        "ip_fast": 0.3159,
        "ip": 0.3259,
        "l2_fast": 0.2420,
        "l2": 0.2469,
    },
    "Sentence\nBERT\n(Avg)": {
        "ip_fast": 0.2586,
        "ip": 0.2650,
        "l2_fast": 0.2893,
        "l2": 0.2958,
    },
    "BERT\n(Avg)": {
        "ip_fast": 0.1451,
        "ip": 0.1460,
        "l2_fast": 0.1970,
        "l2": 0.2084,
    },
    "DPR\n[CLS]": {
        "ip_fast": 0.4029,
        "ip": 0.4391,
        "l2_fast": 0.1745,
        "l2": 0.1845,
    },
    "Sentence\nBERT\n[CLS]": {
        "ip_fast": 0.2599,
        "ip": 0.2687,
        "l2_fast": 0.2610,
        "l2": 0.2691,
    },
    "BERT\n[CLS]": {
        "ip_fast": 0.0042,
        "ip": 0.0025,
        "l2_fast": 0.0291,
        "l2": 0.0313,
    },
}

DATA_RPREC_NORM = {
    "DPR\n(Avg)": {
        "ip_fast": 0.3171,
        "ip": 0.3280,
        "l2_fast": 0.3194,
        "l2": 0.3280,
    },
    "Sentence\nBERT\n(Avg)": {
        "ip_fast": 0.3028,
        "ip": 0.3097,
        "l2_fast": 0.3013,
        "l2": 0.3097,
    },
    "BERT\n(Avg)": {
        "ip_fast": 0.2024,
        "ip": 0.2124,
        "l2_fast": 0.2022,
        "l2": 0.2124,
    },
    "DPR\n[CLS]": {
        "ip_fast": 0.3229,
        "ip": 0.3398,
        "l2_fast": 0.3206,
        "l2": 0.3398,
    },
    "Sentence\nBERT\n[CLS]": {
        "ip_fast": 0.2648,
        "ip": 0.2732,
        "l2_fast": 0.2637,
        "l2": 0.2732,
    },
    "BERT\n[CLS]": {
        "ip_fast": 0.0318,
        "ip": 0.0347,
        "l2_fast": 0.0315,
        "l2": 0.0347,
    },
}

POS = np.arange(len(DATA_RPREC))
BARHEIGHT = 0.2
YERRMOCK = np.random.rand(len(DATA_RPREC))/10000

plt.figure(figsize=(4.4, 4))
ax = plt.gca()
ax.bar(
    POS + 0.2 * 0,
    [x["ip"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="IP",
    color="tab:blue", hatch="",
)
ax.scatter(
    POS + 0.2 * 0,
    [x["ip"] for x in DATA_RPREC_NORM.values()],
    color="black", marker="_"
).set_zorder(10)
ax.bar(
    POS + 0.2 * 1,
    [x["ip_fast"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="IP fast",
    color="lightskyblue", hatch="...",
)
ax.scatter(
    POS + 0.2 * 1,
    [x["ip_fast"] for x in DATA_RPREC_NORM.values()],
    color="black", marker="_"
).set_zorder(10)
ax.bar(
    POS + 0.2 * 2,
    [x["l2"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="L2",
    color="tab:red", hatch="",
)
ax.scatter(
    POS + 0.2 * 2,
    [x["l2"] for x in DATA_RPREC_NORM.values()],
    color="black", marker="_"
).set_zorder(10)
ax.bar(
    POS + 0.2 * 3,
    [x["l2_fast"] for x in DATA_RPREC.values()],
    width=BARHEIGHT, label="L2 fast",
    color="lightcoral", hatch="...",
)
ax.scatter(
    POS + 0.2 * 3,
    [x["l2_fast"] for x in DATA_RPREC_NORM.values()],
    color="black", marker="_"
).set_zorder(10)
ax.set_xticks(POS + 0.15 * 2)
ax.set_xticklabels(DATA_RPREC.keys())
ax.set_ylabel("R-Precision")
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
plt.tight_layout()
plt.show()
