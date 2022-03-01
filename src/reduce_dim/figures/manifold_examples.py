#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

N_VECTORS = 9

data_3 = np.random.RandomState(3).random(N_VECTORS*3).reshape(N_VECTORS, 3)
random_generator = np.random.RandomState(3)

print(data_3)
data_2_p = PCA(n_components=2).fit_transform(data_3)
data_2_t = TSNE(n_components=2, learning_rate="auto", init="pca", n_iter=10000).fit_transform(data_3) / 180
data_2_m = MDS(n_components=2, metric=True).fit_transform(data_3)
data_2_n = MDS(n_components=2, metric=False).fit_transform(data_3)

STYLE = {}

plt.figure(figsize=(4.5, 4))
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
ax.view_init(30, 20)
 
# plot legs
for point in data_3:
    ax.plot3D(
        [point[0], point[0]],
        [point[1], point[1]],
        [0, point[2]],
        color="black",
        alpha=0.5,
        linestyle="dashed",
        linewidth=1,
    )
    # plot points
    ax.scatter3D(
        [point[0]],
        [point[1]],
        [point[2]],
    )

ax.plot3D(
    [data_3[0][0], data_3[8][0], data_3[1][0]],
    [data_3[0][1], data_3[8][1], data_3[1][1]],
    [data_3[0][2], data_3[8][2], data_3[1][2]],
    color="black",
    alpha=1,
    linestyle="solid",
)
plt.tight_layout()
plt.savefig("figures/manifold_examples_3d.pdf")
plt.show()

plt.figure(figsize=(4.5, 4))
plt.scatter(
    [x[0] for x in data_2_p],
    [x[1] for x in data_2_p],
    label="PCA",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data_2_t],
    [x[1] for x in data_2_t],
    label="t-SNE",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data_2_m],
    [x[1] for x in data_2_m],
    label="MDS",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data_2_n],
    [x[1] for x in data_2_n],
    label="MDS (nonmetric)",
    **STYLE,
)


plt.plot(
    [x[0] for x in data_2_p[[0, 8, 1]]],
    [x[1] for x in data_2_p[[0, 8, 1]]],
    color="black", alpha=0.7,
)
plt.plot(
    [x[0] for x in data_2_t[[0, 8, 1]]],
    [x[1] for x in data_2_t[[0, 8, 1]]],
    color="black", alpha=0.7,
)
plt.plot(
    [x[0] for x in data_2_m[[0, 8, 1]]],
    [x[1] for x in data_2_m[[0, 8, 1]]],
    color="black", alpha=0.7,
)
plt.plot(
    [x[0] for x in data_2_n[[0, 8, 1]]],
    [x[1] for x in data_2_n[[0, 8, 1]]],
    color="black", alpha=0.7,
)

plt.legend()
plt.tight_layout()
plt.savefig("figures/manifold_examples_2d.pdf")
plt.show()
