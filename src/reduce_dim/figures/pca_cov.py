#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    data = eval(f.read())

plt.figure(figsize=(4.6, 3.5))
print(data["docs"])
plt.imshow(data["docs"])
plt.tight_layout()
plt.show()