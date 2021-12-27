#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
from model_normalization_data import *

POS = np.arange(len(DATA_BASE))
BARHEIGHT = 0.15
BARSPACE = 0.15
YERRMOCK = np.random.rand(len(DATA_BASE))/10000

plt.figure(figsize=(4.4, 4))
ax = plt.gca()

ax.bar(
    POS + BARSPACE * 0,
    [x["ip"] for x in DATA_BASE.values()],
    width=BARHEIGHT, label="IP",
    color="tab:red", hatch="",
    edgecolor="black",
)
ax.bar(
    POS + BARSPACE * 2,
    [x["ip"] for x in DATA_C.values()],
    width=BARHEIGHT, label="IP (center)",
    color="tab:red", hatch="\\\\\\",
    edgecolor="black",
)
ax.bar(
    POS + BARSPACE * 4,
    [x for x in DATA_N.values()],
    width=BARHEIGHT, label="IP, $L^2$ (norm)",
    color="silver", hatch="...",
    edgecolor="black",
)
ax.bar(
    POS + BARSPACE * 1,
    [x["l2"] for x in DATA_BASE.values()],
    width=BARHEIGHT, label="$L^2$",
    color="tab:blue", hatch="",
    edgecolor="black",
)
ax.bar(
    POS + BARSPACE * 3,
    [x["l2"] for x in DATA_C.values()],
    width=BARHEIGHT, label="$L^2$ (center)",
    color="tab:blue", hatch="\\\\\\",
    edgecolor="black",
)
ax.bar(
    POS + BARSPACE * 5,
    [x for x in DATA_NC.values()],
    width=BARHEIGHT, label="IP, $L^2$ (center, norm)",
    color="dimgray", hatch="...\\\\\\",
    edgecolor="black",
)
ax.set_xticks(POS + BARSPACE * 2.5)
ax.set_xticklabels(DATA_BASE.keys())
ax.set_ylabel(r"R-Precision")
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
plt.tight_layout()
plt.savefig("figures/model_normalization.pdf")
plt.show()
