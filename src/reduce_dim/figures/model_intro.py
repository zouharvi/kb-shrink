#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt


DATA = {
    "DPR\n(avg)": {
        "ip_fast": 0.532,
        "ip": 0.8176,
        "l2_fast": 0.5508,
        "l2": 0.7154,
    },
    "Sentence\nBert\n(avg)": {
        "ip_fast": 0.6268,
        "ip": 0.7284,
        "l2_fast": 0.5976,
        "l2": 0.7304,
    },
    "Bert\n(avg)": {
        "ip_fast": 0.5270,
        "ip": 0.5420,
        "l2_fast": 0.5256,
        "l2": 0.6142,
    },
    "DPR\n(cls)": {
        "ip_fast": 0.461,
        "ip": 0.8862,
        "l2_fast": 0.4682,
        "l2": 0.6036,
    },
    "Sentence\nBert\n(cls)": {
        "ip_fast": 0.5738,
        "ip": 0.7178,
        "l2_fast": 0.555,
        "l2": 0.7036,
    },
    "Bert\n(cls)": {
        "ip_fast": 0.1834,
        "ip": 0.0322,
        "l2_fast": 0.1428,
        "l2": 0.1894,
    },
}

POS = np.arange(len(DATA))
BARHEIGHT = 0.2
YERRMOCK = np.random.rand(len(DATA))/10000

plt.figure(figsize=(4.4, 4))
ax = plt.gca()
ax.bar(
    POS + 0.2 * 0,
    [x["ip"] for x in DATA.values()],
    width=BARHEIGHT, label="IP", yerr=YERRMOCK,
    color="tab:blue", hatch="",
)
ax.bar(
    POS + 0.2 * 1,
    [x["ip_fast"] for x in DATA.values()],
    width=BARHEIGHT, label="IP fast", yerr=YERRMOCK,
    color="lightskyblue", hatch="...",
)
ax.bar(
    POS + 0.2 * 2,
    [x["l2"] for x in DATA.values()],
    width=BARHEIGHT, label="L2", yerr=YERRMOCK,
    color="tab:red", hatch="",
)
ax.bar(
    POS + 0.2 * 3,
    [x["l2_fast"] for x in DATA.values()],
    width=BARHEIGHT, label="L2 fast", yerr=YERRMOCK,
    color="lightcoral", hatch="...",
)
ax.set_xticks(POS + 0.15 * 2)
ax.set_xticklabels(DATA.keys())
ax.set_ylabel("Accuracy (n=20)")
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
plt.tight_layout()
plt.show()
