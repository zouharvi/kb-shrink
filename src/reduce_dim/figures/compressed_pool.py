#!/usr/bin/env python3

import sys
sys.path.append("src")
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(help="Examine how the compression behaves if we reduce the number of data points to compress")
parser.add_argument('--logfile', default="computed/size_perf_128.py")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATARAW = eval(f.read())

DATA = defaultdict(lambda: [])
for line in DATARAW:
    DATA[line["method"]].append(line)

plt.figure(figsize=(6, 4))
ax = plt.gca() 

for method, values in DATA.items():
    if method == "uncompressed":
        continue
    # ax.plot([x["count"] for x in values], [x["rprec_ip"] for x in values], label=method)
    ax.plot(
        list(range(len(values))),
        # [x["count"] for x in values],
        [x["rprec_l2"]/y["rprec_l2"] for x,y in zip(values, DATA["uncompressed"])],
        label=method,
        marker=".",
    )

plt.xticks(
    list(range(len(DATA["uncompressed"]))),
    [x["count"] for x in DATA["uncompressed"]],
    rotation=45,
)
plt.ylabel("Fraction of regained performance (L2)")
plt.xlabel("Number of queries from which documents are gathered")
plt.tight_layout(pad=1.7)
plt.ylim(0.85, 1.0)
plt.legend()
plt.show()