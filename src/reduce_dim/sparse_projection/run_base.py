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
data = read_pickle(args.data)

# take only dev queries
data = sub_data(data, train=False, in_place=True)

# make sure the vectors are np arrays
data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

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
        model = Model(
            n_components=components,
            random_state=random.randint(0, 2**8 - 1)
        )
        model.fit(np.concatenate((data["queries"], data["docs"])))

        dataReduced = {
            "queries": model.transform(data["queries"]),
            "docs": model.transform(data["docs"])
        }
        if args.post_cn:
            dataReduced = center_data(dataReduced)
            dataReduced = norm_data(dataReduced)

        # copy to make it C-continuous
        val_l2 = rprec_a_l2(
            dataReduced["queries"].copy(),
            dataReduced["docs"].copy(),
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
                dataReduced["queries"].copy(),
                dataReduced["docs"].copy(),
                data["relevancy"],
                data["relevancy_articles"],
                data["docs_articles"],
                report=False,
                fast=True,
            )
            vals_ip.append(val_ip)
        else:
            vals_ip.append(val_l2)

    logdata.append({
        "dim": components,
        "vals_ip": vals_ip,
        "vals_l2": vals_l2,
        "model": model_name
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))

DIMS = process_dims(args.dims)
logdata = []

for model in ["crop", "sparse", "gauss"]:
    for dim in DIMS:
        dim = int(dim)
        random_projection_performance(dim, model)
