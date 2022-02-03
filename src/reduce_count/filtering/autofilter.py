#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, small_data, sub_data
from misc.retrieval_utils import retrieved_ip, acc_ip
from filtering_utils import filter_step
import argparse, json, pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd_cn")
parser.add_argument('--logfile', default="computed/autofilter.json")
parser.add_argument('--traindata', default="computed/autofilter_traindata.pkl")
args = parser.parse_args()

data = read_pickle(args.data)
data = sub_data(data, train=True, in_place=True)
data_dev = read_pickle(args.data)
data_dev = sub_data(data_dev, train=False, in_place=True)

del data["relevancy_articles"]
del data["docs_articles"]
# data["queries"] = data["queries"][:24000]
# data["relevancy"] = data["relevancy"][:24000]
del data_dev["docs_articles"]
del data_dev["relevancy_articles"]
del data_dev["docs"]

print(len(data["queries"]), "train queries")
print(len(data_dev["queries"]), "dev queries")

def comp_acc(data, data_dev):
    val_acc = acc_ip(
        data_dev["queries"], data["docs"], data_dev["relevancy"], n=10
    )
    return val_acc


cur_time = time.time()
val_acc = comp_acc(data, data_dev)

logdata_all = []
logdata_all.append({"acc": val_acc})
traindata_all = []

for i in range(10):
    print(f"Pass #### {i+1}", flush=True)
    traindata, logitem = filter_step(data, data_dev, cur_time=cur_time)
    
    cur_time = time.time()

    val_acc = comp_acc(data, data_dev)
    logitem["acc"] = val_acc

    logdata_all.append(logitem)
    traindata_all.append(traindata)

    # continuously overwrite logfile
    with open(args.logfile, "w") as f:
        json.dump(logdata_all, f, indent=4)

    with open(args.traindata, "wb") as f:
        pickle.dump(traindata_all, f)