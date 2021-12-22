#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
from model import transform_to_1, transform_to_8, transform_to_16
import argparse
from sklearn import cluster

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data')
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)


def summary_performance(dataReduced):
    if args.post_cn:
        dataReduced = center_data(dataReduced)
        dataReduced = norm_data(dataReduced)

    val_l2 = rprec_a_l2(
        dataReduced["queries"],
        dataReduced["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        fast=True,
    )
    if args.post_cn:
        val_ip = val_l2
    else:
        val_ip = rprec_a_ip(
            dataReduced["queries"],
            dataReduced["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            fast=True,
        )
    return val_ip, val_l2


def performance_1(data):
    dataReduced = {
        "queries": transform_to_1(data["queries"]),
        "docs": transform_to_1(data["docs"])
    }

    print("Preparing model")
    model = cluster.FeatureAgglomeration(
        n_clusters=384
    )
    print(dataReduced["docs"][0][:10])

    print("Fitting model")
    model.fit(data["docs"])
    dataNew = {
        "docs": model.transform(dataReduced["docs"]),
        "queries": model.transform(dataReduced["queries"]),
    }
    print(dataNew["docs"][0][:10])

    return summary_performance(dataNew)


data = sub_data(data, train=False, in_place=True)

logdata = []
val_ip, val_l2 = performance_1(data)
logdata.append({
    "val_ip": val_ip, "val_l2": val_l2,
    "type": "bit",
})

# continuously override the file
with open(args.logfile, "w") as f:
    f.write(str(logdata))