#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    data = eval(f.read())

data["docs"] = np.array(data["docs"])
data["docs"] = data["docs"][:256,:256]
data["queries"] = np.array(data["queries"])

np.fill_diagonal(data["docs"], 0)

print(np.average(data["docs"]))

plt.figure(figsize=(4.6, 3.5))
print(data["docs"].shape)
plt.imshow(data["docs"])
plt.tight_layout()
plt.show()