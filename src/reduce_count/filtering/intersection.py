#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, small_data, sub_data
from misc.retrieval_utils import retrieved_ip, acc_ip
from filtering_utils import harmful_useful_pass
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd_cn")
args = parser.parse_args()
data = read_pickle(args.data)
data = sub_data(data, train=False, in_place=True)
# data = small_data(data, n_queries=1000)
del data["relevancy_articles"]
del data["docs_articles"]


def comp_acc(data):
    print(f"Running acc on {len(data['docs'])} docs: ", end="")
    val_ip = acc_ip(
        data["queries"], data["docs"], data["relevancy"], n=10
    )
    print(f"{val_ip:.2%}%")


comp_acc(data)

for i in range(3):
    print(f"Pass #### {i+1}")
    data = harmful_useful_pass(data)
    comp_acc(data)
