#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter, defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/hits_redcritter.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    data = eval(f.read())

counts = Counter(zip(data["hits_pca"], data["hits_orig"]))
print(counts)

img = np.zeros((3, 3))
fig = plt.figure(figsize=(3, 2.5))
ax = fig.get_axes()

total = sum(counts.values())
acc_pca = defaultdict(lambda: 0)
acc_orig = defaultdict(lambda: 0)
for ((count_pca, count_orig), count) in counts.items():
    img[count_pca][count_orig] += count
    plt.text(
        x=count_pca,
        y=count_orig,
        s=f"{count/total:.1%}",
        ha="center",
        va="center",
        color="black" if count / total > 0.15 else "white",
    )
    acc_pca[count_pca] += count
    acc_orig[count_orig] += count

# for ((count_pca, count_orig), count) in counts.items():
#     img[count_pca][3] = acc_pca[count_pca]
#     img[3][count_orig] = acc_orig[count_orig]

plt.imshow(
    img,
    aspect=0.9,
)
plt.xlabel("PCA retrieved")
plt.ylabel("Original retrieved")
plt.xticks(range(3))
plt.yticks(range(3))
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
