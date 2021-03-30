#!/usr/bin/env python3

import sys
sys.path.append("src")
import argparse
from misc.utils import mrr, read_keys_pickle, vec_sim_order
from pympler.asizeof import asizeof
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import random
import numpy as np

parser = argparse.ArgumentParser(description='Explore vector distribution')
parser.add_argument('--keys-in', default="data/eli5-dev.embd")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
data = read_keys_pickle(args.keys_in)
origSize = asizeof(data)

order_old = vec_sim_order(data)

print(f"{'Method':<14} {'Size':<13} {'MRR20':<0}")


def summary_performance(name, dataReduced):
    order_new = vec_sim_order(dataReduced)
    mrr_val = mrr(order_old, order_new, 20, report=False)
    size = asizeof(dataReduced)
    print(f"{name:<12} {size/1024/1024:>4.1f}MB ({size/origSize:.1f}x)   {mrr_val:>5.3f}")

def summary_performance_custom(name, dataReduced, mrr_val, mrr_val_avg):
    size = asizeof(dataReduced)
    print(f"{name:<12} {size/1024/1024:>4.1f}MB ({size/origSize:.1f}x)   {mrr_val:>5.3f}")

class CropRandomProjection():
    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random = random.Random(random_state)
    
    def fit_transform(self, data):
        indicies = self.random.sample(range(data[0].shape[0]), k=self.n_components)
        return data.take(indicies, axis=1)

def random_projection_performance(components, model_name, runs=10):
    if model_name == "gauss":
        Model = GaussianRandomProjection
    elif model_name == "sparse":
        Model = SparseRandomProjection
    elif model_name == "crop":
        Model = CropRandomProjection
    else:
        raise Exception("Unknown model")

    random.seed(args.seed)
    mrr_vals = []
    for i in range(runs):
        dataReduced = Model(
            n_components=components,
            random_state=random.randint(0, 2**8-1)
        ).fit_transform(data).astype("float32")
        order_new = vec_sim_order(dataReduced)
        mrr_val = mrr(order_old, order_new, 20, report=False)    
        mrr_vals.append(mrr_val)

    summary_performance_custom(f"{model_name.capitalize()} ({components})", dataReduced, max(mrr_vals), np.average(mrr_vals))


# summary_performance(f"Original ({data.dtype})", data)
random_projection_performance(16, "crop")
random_projection_performance(32, "crop")
random_projection_performance(64, "crop")
random_projection_performance(128, "crop")
random_projection_performance(256, "crop")
random_projection_performance(512, "crop")
# random_projection_performance(16, "sparse")
# random_projection_performance(32, "sparse")
# random_projection_performance(64, "sparse")
# random_projection_performance(128, "sparse")
# random_projection_performance(256, "sparse")
# random_projection_performance(512, "sparse")
# random_projection_performance(16, "gauss")
# random_projection_performance(32, "gauss")
# random_projection_performance(64, "gauss")
# random_projection_performance(128, "gauss")
# random_projection_performance(256, "gauss")
# random_projection_performance(512, "gauss")