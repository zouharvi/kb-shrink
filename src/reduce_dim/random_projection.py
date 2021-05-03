#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import acc_ip_fast, acc_l2_fast, read_keys_pickle
import numpy as np
import random
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from pympler.asizeof import asizeof
import argparse

parser = argparse.ArgumentParser(
    description='Random projection performance summary')
parser.add_argument('--keys-in', default="data/eli5-dev.embd")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
data = read_keys_pickle(args.keys_in)
origSize = asizeof(data)

print(f"{'':<19} {'IPACC':<11} {'L2ACC':<0}")
print(f"{'Method':<12} {'Size':<6} {'(max)|(avg)':<0} {'(max)|(avg)':<0}")


def summary_performance(name, dataReduced):
    acc_val_ip = acc_ip_fast(data, dataReduced, 20, report=False)
    acc_val_l2 = acc_l2_fast(data, dataReduced, 20, report=False)
    size = asizeof(dataReduced)
    print(f"{name:<12} {size/origSize:>5.3f}x {acc_val_ip:>5.3f}|{acc_val_ip:>5.3f} {acc_val_l2:>5.3f}|{acc_val_l2:>5.3f}")


def summary_performance_custom(name, dataReduced, acc_val_ip, acc_avg_ip, acc_val_l2, acc_avg_l2):
    size = asizeof(dataReduced)
    print(f"{name:<12} {size/origSize:>5.3f}x {acc_val_ip:>5.3f}|{acc_avg_ip:>5.3f} {acc_val_l2:>5.3f}|{acc_avg_l2:>5.3f}")


class CropRandomProjection():
    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random = random.Random(random_state)

    def fit_transform(self, data):
        indicies = self.random.sample(
            range(data[0].shape[0]), k=self.n_components)
        return data.take(indicies, axis=1)


def random_projection_performance(components, model_name, runs=5):
    if model_name == "gauss":
        Model = GaussianRandomProjection
    elif model_name == "sparse":
        Model = SparseRandomProjection
    elif model_name == "crop":
        Model = CropRandomProjection
    else:
        raise Exception("Unknown model")

    random.seed(args.seed)
    acc_vals_ip = []
    acc_vals_l2 = []
    for i in range(runs):
        dataReduced = Model(
            n_components=components,
            random_state=random.randint(0, 2**8 - 1)
        ).fit_transform(data).astype("float32")
        # copy to make it C-continuous
        acc_val_ip = acc_ip_fast(data, dataReduced.copy(), 20, report=False)
        acc_vals_ip.append(acc_val_ip)
        acc_val_l2 = acc_l2_fast(data, dataReduced, 20, report=False)
        acc_vals_l2.append(acc_val_l2)

    summary_performance_custom(
        f"{model_name.capitalize()} ({components})", dataReduced,
        max(acc_vals_ip), np.average(acc_vals_ip),
        max(acc_vals_l2), np.average(acc_vals_l2)
    )


summary_performance(f"Original", data)
random_projection_performance(16, "crop")
random_projection_performance(32, "crop")
random_projection_performance(64, "crop")
random_projection_performance(128, "crop")
random_projection_performance(256, "crop")
random_projection_performance(16, "sparse")
random_projection_performance(32, "sparse")
random_projection_performance(64, "sparse")
random_projection_performance(128, "sparse")
random_projection_performance(256, "sparse")
random_projection_performance(16, "gauss")
random_projection_performance(32, "gauss")
random_projection_performance(64, "gauss")
random_projection_performance(128, "gauss")
random_projection_performance(256, "gauss")
