#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_keys_pickle, mrr_ip_fast, mrr_l2_fast
import argparse
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding

parser = argparse.ArgumentParser(description='Visualization of embeddings')
parser.add_argument(
    '--keys', default="data/hotpot-dpr.embd")
args = parser.parse_args()

data = read_keys_pickle(args.keys)[:5000]
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

mrr_l2_fast(data, data_new, report=True)
mrr_ip_fast(data, data_new, report=True)