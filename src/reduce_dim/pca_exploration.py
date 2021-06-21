#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

model = PCA(
    n_components=64,
    random_state=args.seed
).fit(np.concatenate((data["queries"], data["docs"])))

print(np.matmul(model.components_, model.components_.T) >= 0.01)

# dataReduced = {
#     "queries": model.transform(data["queries"]),
#     "docs": model.transform(data["docs"])
# }
# val_ip_pca = rprec_ip(
#     dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
# )
# val_l2_pca = rprec_l2(
#     dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
# )
