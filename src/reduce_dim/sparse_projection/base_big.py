#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
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
            range(_data["queries"][0].shape[0]),
            k=self.n_components
        )

data_log = []

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
        data = read_pickle(args.data)
        # take only dev queries
        data = sub_data(data, train=False, in_place=True)
        # make sure the vectors are np arrays
        data["queries"] = np.array(data["queries"])
        data["docs"] = np.array(data["docs"])

        model = Model(
            n_components=components,
            random_state=random.randint(0, 2**8 - 1)
        )
        model.fit(data["docs"])

        dataReduced = {
            "queries": model.transform(data["queries"]),
            "docs": model.transform(data["docs"])
        }
        del data["queries"]
        del data["docs"]

        if args.post_cn:
            dataReduced = center_data(dataReduced)
            dataReduced = norm_data(dataReduced)

        # copy to make it C-continuous
        val_l2 = rprec_a_l2(
            np.ascontiguousarray(dataReduced["queries"]),
            np.ascontiguousarray(dataReduced["docs"]),
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            report=False,
            fast=True,
        )
        vals_l2.append(val_l2)

        # skip IP computation because the vectors are normalized
        if not args.post_cn:
            val_ip = rprec_a_ip(
                np.ascontiguousarray(dataReduced["queries"]),
                np.ascontiguousarray(dataReduced["docs"]),
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


if args.dims == "custom":
    DIMS = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
elif args.dims == "linspace":
    DIMS = np.linspace(32, 768, num=768 // 32, endpoint=True)
else:
    raise Exception(f"Unknown --dims {args.dims} scheme")

for dim in [640, 768]:
    random_projection_performance(dim, "sparse")

for dim in DIMS:
    random_projection_performance(dim, "gauss")
