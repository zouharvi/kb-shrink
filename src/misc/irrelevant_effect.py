#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--step', type=int, default=10000)
parser.add_argument('--n', default=5000, help="Queries base")
args = parser.parse_args()
data = read_pickle(args.data)

data["queries"] = data["queries"][:args.n]
data["relevancy"] = data["relevancy"][:args.n]
max_base_doc = max([x for rel_l in data["relevancy"] for x in rel_l])
print(f"Making {(len(data['docs'])-max_base_doc)//args.step} steps from {max_base_doc} (base) to {len(data['docs'])} (total doc count)")

logdata = []

for threshold in range(max_base_doc, len(data['docs'])+args.step-1, args.step):
    data_cropped = data["docs"][:threshold]
    val_ip = rprec_ip(
        data["queries"], data_cropped, data["relevancy"], fast=True
    )
    val_l2 = rprec_l2(
        data["queries"], data_cropped, data["relevancy"], fast=True
    )
    print(f"threshold: {threshold}, ip: {val_ip:.4f}, l2: {val_l2:.4f}")
    logdata.append({"threshold": threshold, "val_ip": val_ip, "val_l2": val_l2})

    # override dump
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
