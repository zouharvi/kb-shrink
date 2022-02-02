#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, small_data, sub_data
from misc.retrieval_utils import retrieved_ip, acc_ip
from filtering_utils import filter_step
import argparse, json, pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd_cn")
parser.add_argument('--logfile', default="computed/autofilter.json")
parser.add_argument('--traindata', default="computed/autofilter_traindata.pkl")
args = parser.parse_args()
data = read_pickle(args.data)
data = sub_data(data, train=True, in_place=True)
# data = small_data(data, n_queries=50)
del data["relevancy_articles"]
del data["docs_articles"]

def comp_acc(data):
    val_acc = acc_ip(
        data["queries"], data["docs"], data["relevancy"], n=10
    )
    return val_acc


val_acc = comp_acc(data)

logdata_all = []
logdata_all.append({"acc": val_acc})
traindata_all = []

for i in range(10):
    print(f"Pass #### {i+1}")
    traindata, logitem = filter_step(data)

    val_acc = comp_acc(data)
    logitem["acc"] = val_acc

    logdata_all.append(logitem)
    traindata_all.append(traindata)

    # continuously overwrite logfile
    with open(args.logfile, "w") as f:
        json.dump(logdata_all, f, indent=4)

    with open(args.traindata, "wb") as f:
        pickle.dump(traindata_all, f)