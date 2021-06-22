#!/usr/bin/env python3

import argparse
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import random
import numpy as np
from misc.utils import rprec_ip, rprec_l2, read_pickle
import sys
sys.path.append("src")

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
data = read_pickle(args.data)
data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

print(f"{'':<12} {'IP':<12} {'L2':<12}")
print(f"{'Method':<12} {'(max)|(avg)':<12} {'(max)|(avg)':<12}")


def summary_performance_custom(name, acc_val_ip, acc_avg_ip, acc_val_l2, acc_avg_l2):
    print(f"{name:<12} {acc_val_ip:>5.3f}|{acc_avg_ip:>5.3f} {acc_val_l2:>5.3f}|{acc_avg_l2:>5.3f}")


class CropRandomProjection():
    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random = random.Random(random_state)
        self.indicies = None

    def transform(self, data):
        return np.array(data).take(self.indicies, axis=1)

    def fit(self, _data):
        self.indicies = self.random.sample(
            range(data["queries"][0].shape[0]),
            k=self.n_components
        )


data_log = []


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
    vals_ip = []
    vals_l2 = []
    for i in range(runs):
        model = Model(
            n_components=components,
            random_state=random.randint(0, 2**8 - 1)
        )
        model.fit(np.concatenate((data["queries"], data["docs"])))

        dataReduced = {
            "queries": model.transform(data["queries"]),
            "docs": model.transform(data["docs"])
        }

        # copy to make it C-continuous
        val_ip = rprec_ip(
            dataReduced["queries"].copy(), dataReduced["docs"].copy(), data["relevancy"], report=False, fast=True
        )
        vals_ip.append(val_ip)
        val_l2 = rprec_l2(
            dataReduced["queries"].copy(), dataReduced["docs"].copy(), data["relevancy"], report=False, fast=True
        )
        vals_l2.append(val_l2)

    data_log.append({"dim": components, "vals_ip": vals_ip,
                    "vals_l2": vals_l2, "model": model_name})

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

    summary_performance_custom(
        f"{model_name.capitalize()} ({components})",
        max(vals_ip), np.average(vals_ip),
        max(vals_l2), np.average(vals_l2)
    )


for model in ["crop", "sparse", "gauss"]:
    for dim in np.linspace(32, 768, num=768//32, endpoint=True):
        dim = int(dim)
        random_projection_performance(dim, model)
