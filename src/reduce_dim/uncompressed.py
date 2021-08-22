#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import acc_ip, acc_l2, read_pickle, rprec_l2, rprec_ip, rprec_n_l2, rprec_n_ip, center_data, norm_data
import argparse

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data', default="data/hotpot.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--all', action="store_true")
parser.add_argument('--metric', default="rprec")
parser.add_argument('--train', action="store_true")
parser.add_argument('--without-faiss', action="store_true")
args = parser.parse_args()
data = read_pickle(args.data)
if args.train:
    data["queries"] = data["queries"][:data["boundaries"]["train"]]
    data["relevancy"] = data["relevancy"][:data["boundaries"]["train"]]
else:
    # default is dev only
    data["queries"] = data["queries"][data["boundaries"]["train"]:data["boundaries"]["dev"]]
    data["relevancy"] = data["relevancy"][data["boundaries"]["train"]:data["boundaries"]["dev"]]

if args.metric == "rprec":
    metric_l2 = rprec_l2
    metric_ip = rprec_ip 
elif args.metric == "rprec_n":
    metric_l2 = rprec_n_l2
    metric_ip = rprec_n_ip 
elif args.metric == "acc":
    metric_l2 = acc_l2
    metric_ip = acc_ip
else:
    raise Exception("Unknown metric")

assert not args.all or not (args.center or args.norm)


if args.all:
    data_b = data
    print(f"{args.metric}_ip:", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print(f"{args.metric}_l2:", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))

    data = norm_data(data_b)
    print(f"{args.metric}_ip (norm):", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print(f"{args.metric}_l2 (norm):", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    
    data = center_data(data_b)
    print(f"{args.metric}_ip (center):", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print(f"{args.metric}_l2 (center):", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    
    data = norm_data(center_data(data_b))
    print(f"{args.metric}_ip (center, norm):", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print(f"{args.metric}_l2 (center, norm):", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
else:
    if args.center:
        data = center_data(data)
    if args.norm:
        data = norm_data(data)

    # RPrec
    print(f"{args.metric}_ip (fast)", metric_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    ))
    if args.without_faiss:
        print(f"{args.metric}_ip", metric_ip(
            data["queries"], data["docs"], data["relevancy"]
        ))
    print(f"{args.metric}_l2 (fast)", metric_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    ))
    if args.without_faiss:
        print(f"{args.metric}_l2", metric_l2(
            data["queries"], data["docs"], data["relevancy"]
        ))