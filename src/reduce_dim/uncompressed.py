#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_pickle, rprec_l2, rprec_ip, center_data, norm_data
import argparse

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data', default="data/hotpot.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--all', action="store_true")
args = parser.parse_args()
data = read_pickle(args.data)

assert not args.all or not (args.center or args.norm)

if args.all:
    data_b = data
    print("rprec_ip_fast:", "{:.4f}".format(rprec_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print("rprec_l2_fast:", "{:.4f}".format(rprec_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))

    data = norm_data(data_b)
    print("rprec_ip_fast (norm):", "{:.4f}".format(rprec_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print("rprec_l2_fast (norm):", "{:.4f}".format(rprec_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    
    data = center_data(data_b)
    print("rprec_ip_fast (center):", "{:.4f}".format(rprec_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print("rprec_l2_fast (center):", "{:.4f}".format(rprec_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    
    data = norm_data(center_data(data_b))
    print("rprec_ip_fast (center, norm):", "{:.4f}".format(rprec_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
    print("rprec_l2_fast (center, norm):", "{:.4f}".format(rprec_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    )))
else:
    if args.center:
        data = center_data(data)
    if args.norm:
        data = norm_data(data)

    # RPrec
    print("rprec_ip_fast", rprec_ip(
        data["queries"], data["docs"], data["relevancy"], fast=True
    ))
    print("rprec_ip", rprec_ip(
        data["queries"], data["docs"], data["relevancy"]
    ))
    print("rprec_l2_fast", rprec_l2(
        data["queries"], data["docs"], data["relevancy"], fast=True
    ))
    print("rprec_l2", rprec_l2(
        data["queries"], data["docs"], data["relevancy"]
    ))