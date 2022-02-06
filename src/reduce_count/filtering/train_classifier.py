#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, save_json, small_data, sub_data
from misc.retrieval_utils import retrieved_ip, acc_ip
from filtering_utils import filter_step, prune_docs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd_cn")
parser.add_argument('--logfile', default="computed/autofilter_classifier.json")
parser.add_argument('--traindata', default="computed/autofilter_traindata.pkl")
args = parser.parse_args()
data_train = read_pickle(args.traindata)

logdata = []
positives_total = []
negatives_total = []

def eval_retrieval(scaler, model):
    # This loads the data from disk every time
    # To speed this up, we could load this once and then "only" make copies
    # The reason we can't have a single copy is that the model predictions can differ
    data = read_pickle(args.data)
    data = sub_data(data, train=False, in_place=True)
    prediction = model.predict(scaler.transform(data["docs"]))

    data = prune_docs(
        data, None,
        [i for i, _x in enumerate(data["docs"]) if prediction[i] == 1],
        verbose=False
    )
    
    acc = acc_ip(
        data["queries"], data["docs"], data["relevancy"], n=10
    )
    return acc

for step, data_step in enumerate(data_train):
    logitem = {}
    print(f"\n### Step {step:>02}")
    print(f"{len(data_step['positive'])} positives")
    print(f"{len(data_step['negative'])} negatives")
    print(f"{len(data_step['positive'])/(len(data_step['negative'])+len(data_step['positive'])):.2%}% positives")
    logitem["positive"] = len(data_step['positive'])
    logitem["negative"] = len(data_step['negative'])
    
    data_local_x = data_step["positive"] + data_step["negative"]
    scaler = StandardScaler()
    data_local_x = scaler.fit_transform(data_local_x)
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

    # logitem["acc"] = eval_retrieval(scaler, model)

    positives_total += data_step["positive"]
    negatives_total += data_step["negative"]

    data_all_x = positives_total + negatives_total
    scaler = StandardScaler()
    data_all_x = scaler.fit_transform(data_all_x)
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

    # logitem["acc_total"] = eval_retrieval(scaler, model)

    logdata.append(logitem)
    save_json(args.logfile, logdata)