#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_ip, rprec_a_l2
import argparse
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--dims', default="custom")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

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
        return data.take(self.indicies, axis=1)

    def fit(self, _data):
        self.indicies = self.random.sample(
            range(_data[0].shape[0]),
            k=self.n_components
        )

def safe_print(msg):
    with open("base_big_rproj.out", "a") as f:
        f.write(msg+"\n")

data_log = [] 

def safe_transform(model, array):
    return [model.transform(np.array([x]))[0] for x in array]

def random_projection_performance(components, model_name, runs=3):
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
        safe_print(" ")
        safe_print("A")
        data = read_pickle(args.data)
        # take only dev queries
        data = sub_data(data, train=False, in_place=True)
        safe_print("B")
        # make sure the vectors are np arrays
        data["queries"] = np.array(data["queries"])
        safe_print("C")
        data["docs"] = np.array(data["docs"])
        safe_print("D")

        model = Model(
            n_components=components,
            random_state=random.randint(0, 2**8 - 1)
        )
        model.fit(data["docs"])
        safe_print("E")

        dataReduced = {
            "queries": safe_transform(model, data["queries"]),
            "docs": safe_transform(model, data["docs"])
        }
        safe_print("F")
        del data["queries"]
        del data["docs"]
        safe_print("G")

        if args.post_cn:
            dataReduced = center_data(dataReduced)
            dataReduced = norm_data(dataReduced)
        safe_print("H")

        # copy to make it C-continuous
        # (skipped)
        val_l2 = rprec_a_l2(
            dataReduced["queries"],
            dataReduced["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            report=False,
            fast=True,
        )
        vals_l2.append(val_l2)
        safe_print("I")

        # skip IP computation because the vectors are normalized
        if not args.post_cn:
            val_ip = rprec_a_ip(
                dataReduced["queries"],
                dataReduced["docs"],
                data["relevancy"],
                data["relevancy_articles"],
                data["docs_articles"],
                report=False,
                fast=True,
            )
            vals_ip.append(val_ip)
        else:
            vals_ip.append(val_l2)

    data_log.append({
        "dim": components,
        "vals_ip": vals_ip,
        "vals_l2": vals_l2,
        "model": model_name
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(data_log))

    summary_performance_custom(
        f"{model_name.capitalize()} ({components})",
        max(vals_ip), np.average(vals_ip),
        max(vals_l2), np.average(vals_l2)
    )

DIMS = process_dims(args.dims)

for dim in DIMS:
    random_projection_performance(dim, "crop")
    random_projection_performance(dim, "sparse")
    random_projection_performance(dim, "gauss")
