#!/usr/bin/env python3

import copy
import random
import sys
sys.path.append("src")
from misc.load_utils import read_pickle, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import argparse

parser = argparse.ArgumentParser(description='Uncompressed irrelevant performance summary')
parser.add_argument('--data')
parser.add_argument('--data-big')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
data_big = read_pickle(args.data_big)

def summary_performance(dataReduced):
    val_l2 = rprec_a_l2(
        dataReduced["queries"],
        dataReduced["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        fast=True,
    )
    val_ip = val_l2
    return val_ip, val_l2

data = sub_data(data, train=False, in_place=True)

logdata = []
# for num_samples in [(10**6) * 3]:
# for num_samples in [10**7]:
# for num_samples in [(10**7) * 3]:
for num_samples in [(10**6) * 3, 10**7, (10**7) * 3]:
    # increase test size
    new_data = copy.deepcopy(data)
    new_data["docs"] += random.sample(
        data_big["docs"],
        num_samples - len(data["docs"])
    )
    val_ip, val_l2 = summary_performance(new_data)
    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "num_samples": num_samples,
        "type": "eval_data",
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
