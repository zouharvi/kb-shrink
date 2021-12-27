#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
from model import transform_to_1
import argparse
from sklearn.decomposition import PCA

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
    model = PCA(n_components=256)
    model.fit(data["docs"])
    dataReduced = {
        "docs": model.transform(data["docs"]),
        "queries": model.transform(data["queries"]),
    }

    # center data in between
    # actually worsesns the performance slightly
    # dataReduced = center_data(dataReduced)
    # dataReduced = norm_data(dataReduced)

    dataReduced = {
        "queries": transform_to_1(dataReduced["queries"]),
        "docs": transform_to_1(dataReduced["docs"])
    }

    return summary_performance(dataReduced)


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