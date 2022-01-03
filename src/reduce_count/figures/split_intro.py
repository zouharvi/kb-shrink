#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(4.4, 4))
ax = plt.gca()
ax.scatter(
    0, 1,
    label="IP",
    color="tab:red", hatch="",
    edgecolor="black",
)
ax.set_ylabel("R-Precision")
ax.set_xlabel("Passage count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/split_intro.pdf")
plt.show()