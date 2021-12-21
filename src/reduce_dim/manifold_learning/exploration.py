#!/usr/bin/env python3

raise NotImplementedError(
    "Not adapted to new data orgnization (docs and queries as tuples)")

import sys
sys.path.append("src")
from misc.load_utils import read_keys_pickle
from misc.retrieval_utils import acc_ip, acc_l2
import argparse
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding

parser = argparse.ArgumentParser(description='Visualization of embeddings')
parser.add_argument(
    '--keys', default="data/hotpot-dpr.embd")
args = parser.parse_args()

data = read_keys_pickle(args.keys)
# model = TSNE(n_components=3, random_state=0)

# mds
model = MDS(n_components=64)
data_new = model.fit_transform(data)

# isomap
# model = Isomap(n_components=64)
# data_new = model.fit_transform(data)
# data_new = data_new.copy() # is not C-continuous

# spectral embedding
# model = SpectralEmbedding(n_components=64)
# data_new = model.fit_transform(data)
# data_new = data_new.copy() # is not C-continuous

acc_l2(data, data_new, report=True)
acc_ip(data, data_new, report=True)
