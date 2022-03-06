#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
from misc.load_utils import read_pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

N_VECTORS = 15
# INDEX_0 = 2
# INDEX_1 = 1
INDEX_0 = 2
INDEX_1 = 3

def center(data):
    return data - data.mean(axis=0)


def norm(data):
    return data / np.linalg.norm(data, axis=1)[:, np.newaxis]


# data = np.random.RandomState(3).random(N_VECTORS*3).reshape(N_VECTORS, 3)
data_q = np.array(read_pickle("computed/dpr-c-pruned-500.embd")["queries"])[:,[INDEX_0,INDEX_1]]
data_d = np.array(read_pickle("computed/dpr-c-pruned-500.embd")["docs"])[:,[INDEX_0,INDEX_1]]

data_q_c = center(data_q)[:N_VECTORS]
data_q_cn = norm(center(data_q))[:N_VECTORS] * 1
data_q_n = norm(data_q)[:N_VECTORS]
data_d_c = center(data_d)[:N_VECTORS]
data_d_cn = norm(center(data_d))[:N_VECTORS] * 1
data_d_n = norm(data_d)[:N_VECTORS]
data_q = data_q[:N_VECTORS]
data_d = data_d[:N_VECTORS]

STYLE = {"alpha": 0.7, "s": 15}
STYLE_D = {"marker": "s"}

plt.figure(figsize=(6, 4))
h_q = plt.scatter(
    [x[0] for x in data_q],
    [x[1] for x in data_q],
    label="Original",
    color="tab:gray",
    **STYLE,
)
h_q_c = plt.scatter(
    [x[0] for x in data_q_c],
    [x[1] for x in data_q_c],
    label="Center",
    color="tab:red",
    **STYLE,
)
h_q_cn = plt.scatter(
    [x[0] for x in data_q_cn],
    [x[1] for x in data_q_cn],
    label="Center+Norm",
    color="tab:blue",
    **STYLE,
)
# plt.scatter(
#     [x[0] for x in data_q_n],
#     [x[1] for x in data_q_n],
#     label="Norm",
#     color="tab:green",
#     **STYLE,
# )

h_d = plt.scatter(
    [x[0] for x in data_d],
    [x[1] for x in data_d],
    label="Original",
    color="tab:gray",
    **STYLE, **STYLE_D,
)
h_d_c = plt.scatter(
    [x[0] for x in data_d_c],
    [x[1] for x in data_d_c],
    label="Center",
    color="tab:red",
    **STYLE, **STYLE_D,
)
h_d_cn = plt.scatter(
    [x[0] for x in data_d_cn],
    [x[1] for x in data_d_cn],
    label="Center+Norm",
    color="tab:blue",
    **STYLE, **STYLE_D,
)
# plt.scatter(
#     [x[0] for x in data_d_n],
#     [x[1] for x in data_d_n],
#     label="Norm",
#     color="tab:green",
#     **STYLE, **STYLE_D,
# )

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                  edgecolor='none', linewidth=0)

plt.legend(
    [extra, h_q, h_q_c, h_q_cn, extra, h_d, h_d_c, h_d_cn],
    [
        "$\\textbf{Queries}$", h_q.get_label(), h_q_c.get_label(), h_q_cn.get_label(),
        "$\\textbf{Docs}$", h_d.get_label(), h_d_c.get_label(), h_d_cn.get_label()
    ],
    bbox_to_anchor=(1, 1), loc="upper left"
)
plt.tight_layout()
plt.savefig("figures/norm_examples.pdf")
plt.show()
