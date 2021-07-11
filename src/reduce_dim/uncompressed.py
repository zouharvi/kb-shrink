#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import acc_ip, acc_l2, read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data', default="data/hotpot.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--all', action="store_true")
parser.add_argument('--metric', default="rprec")
args = parser.parse_args()
data = read_pickle(args.data)

if args.metric == "rprec":
    metric_l2 = rprec_l2
    metric_ip = rprec_ip 
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
    print(f"{args.metric}_ip", rprec_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    ))
    print("rprec_ip", rprec_ip(
        data["queries"], data["docs"], data["relevancy"]
    ))
    print(f"{args.metric}_l2", rprec_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    ))
    print("rprec_l2", rprec_l2(
        data["queries"], data["docs"], data["relevancy"]
    ))