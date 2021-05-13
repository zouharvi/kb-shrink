#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt


DATA = {
    "Sentence\nBert\n(tok)": {
        "ip": 0.7324,
        "ip_fast": 0.5930,
        "l2": 0.7310,
        "l2_fast": 0.5676,
    },
    "Bert\n(tok)": {
        "ip": 0.5420,
        "l2": 0.6142,
        "ip_fast": 0.5270,
        "l2_fast": 0.5256,
    },
    "DPR\n(tok)": {
        "ip": 0.0044,
        "ip_fast": 0.0024,
        "l2": 0.004,
        "l2_fast": 0.004,
    },
    "Sentence\nBert\n(pooln)": {
        "ip": 0.2438,
        "ip_fast": 0.2112,
        "l2": 0.2404,
        "l2_fast": 0.2052,
    },
    "Bert\n(pooln)": {
        "ip": 0.0246,
        "ip_fast": 0.0248,
        "l2": 0.0246,
        "l2_fast": 0.0240,
    },
    "DPR\n(pooln)": {
        "ip": 0.0036,
        "ip_fast": 0.0034,
        "l2": 0.0036,
        "l2_fast": 0.0048,
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
# ax.invert_yaxis()
ax.set_xticks(POS + 0.15 * 2)
ax.set_xticklabels(DATA.keys())
ax.set_ylabel("Accuracy (n=20)")
plt.tight_layout()
plt.legend()
plt.show()
