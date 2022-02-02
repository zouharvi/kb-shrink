#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, small_data, sub_data
from misc.retrieval_utils import retrieved_ip, acc_ip
from filtering_utils import filter_step
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/intersection_classifier.log")
parser.add_argument('--traindata', default="computed/intersection_traindata.pkl")
args = parser.parse_args()
data = read_pickle(args.traindata)

logdata = []
positives_total = []
negatives_total = []

for step, data_step in enumerate(data):
    logitem = {}
    print(f"\n### Step {step:>02}")
    print(f"{len(data_step['positive'])} positives")
    print(f"{len(data_step['negative'])} negatives")
    print(f"{len(data_step['positive'])/(len(data_step['negative'])+len(data_step['positive'])):.2%}% positives")
    logitem["positive"] = len(data_step['positive'])
    logitem["negative"] = len(data_step['negative'])
    
    data_local_x = data_step["positive"] + data_step["negative"]
    data_local_x = StandardScaler().fit_transform(data_local_x)
    data_local_y = [1]*len(data_step["positive"]) + [0]*len(data_step["negative"])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(data_local_x, data_local_y)
    acc = model.score(data_local_x, data_local_y)
    print(f"MCCC acc: {acc:.2%}%")
    logitem["mccc"] = acc

    model = LogisticRegression(max_iter=1000)
    model.fit(data_local_x, data_local_y)
    acc = model.score(data_local_x, data_local_y)
    print(f"LR acc: {acc:.2%}%")
    logitem["lr"] = acc

    positives_total += data_step["positive"]
    negatives_total += data_step["negative"]

    data_all_x = positives_total + negatives_total
    data_all_x = StandardScaler().fit_transform(data_all_x)
    data_all_y = [1]*len(positives_total) + [0]*len(negatives_total)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(data_all_x, data_all_y)
    acc = model.score(data_all_x, data_all_y)
    print(f"Total MCCC acc: {acc:.2%}%")
    logitem["mccc_total"] = acc

    model = LogisticRegression(max_iter=1000)
    model.fit(data_all_x, data_all_y)
    acc = model.score(data_all_x, data_all_y)
    print(f"Total LR acc: {acc:.2%}%")
    logitem["lr_total"] = acc

    logdata.append(logitem)

    # continuously overwrite logfile
    with open(args.logfile, "w") as f:
        json.dump(logdata, f, indent=4)