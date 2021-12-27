#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter, defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/hits_pca_redcritter.log")
parser.add_argument('--hide-y', action="store_true")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    data = eval(f.read())

counts = Counter(zip(data["hits_new"], data["hits_old"]))
print(counts)

img = np.zeros((
    max(x for x, y in counts.keys()) + 1,
    max(y for x, y in counts.keys()) + 1
))
fig = plt.figure(figsize=(1.9 if args.hide_y else 2.5, 2.5))
ax = fig.get_axes()

total = sum(counts.values())
acc_new = defaultdict(lambda: 0)
acc_old = defaultdict(lambda: 0)
for ((count_new, count_old), count) in counts.items():
    img[count_new][count_old] += count
    plt.text(
        x=count_new,
        y=count_old,
        s=f"{count/total:.1%}",
        ha="center",
        va="center",
        color="black" if count / total > 0.15 else "white",
    )
    acc_new[count_new] += count
    acc_old[count_old] += count

plt.imshow(
    img,
    # aspect=0.9,
)
if not args.hide_y: 
    plt.ylabel("Original retrieved")
    plt.xlabel("PCA retrieved")
else:
    plt.gca().get_yaxis().set_visible(False)
    plt.xlabel("1bit retrieved")
    
plt.xticks(range(3))
plt.yticks(range(3))
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
