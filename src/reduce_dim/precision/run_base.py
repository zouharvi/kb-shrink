#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
from model import transform_to_1, transform_to_8, transform_to_16
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data')
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)


def summary_performance(dataReduced):
    dataReduced["queries"] = np.array(dataReduced["queries"], dtype=np.float64)
    dataReduced["docs"] = np.array(dataReduced["docs"], dtype=np.float64)
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


def safe_transform(transform, array):
    return [transform(x) for x in array]


def bit_performance_16(data):
    dataReduced = {
        "queries": transform_to_16(data["queries"]),
        "docs": transform_to_16(data["docs"])
    }
    return summary_performance(dataReduced)


def bit_performance_8(data):
    dataReduced = {
        "queries": transform_to_8(data["queries"]),
        "docs": transform_to_8(data["docs"])
    }
    return summary_performance(dataReduced)

def bit_performance_1(data):
    dataReduced = {
        "queries": transform_to_1(data["queries"], offset=0),
        "docs": transform_to_1(data["docs"], offset=0)
    }
    return summary_performance(dataReduced)


data = sub_data(data, train=False, in_place=True)

logdata = []
val_ip, val_l2 = bit_performance_1(data)
logdata.append({
    "val_ip": val_ip, "val_l2": val_l2,
    "type": "bit",
})
# val_ip, val_l2 = bit_performance_8(data)
# logdata.append({
#     "val_ip": val_ip, "val_l2": val_l2,
#     "type": "float8",
# })
# val_ip, val_l2 = bit_performance_16(data)
# logdata.append({
#     "val_ip": val_ip, "val_l2": val_l2,
#     "type": "float16",
# })

# continuously override the file
with open(args.logfile, "w") as f:
    f.write(str(logdata))