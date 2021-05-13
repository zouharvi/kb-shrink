#!/usr/bin/env python3

import sys
import numpy as np
import torch
sys.path.append("src")
from misc.utils import read_pickle, acc_l2, acc_ip, rprec_l2, rprec_ip
import argparse

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data', default="data/hotpot.embd")
args = parser.parse_args()
data = read_pickle(args.data)

# ACC
# print("acc_ip_fast", acc_ip(
#     data["queries"], data["docs"], data["relevancy"], 20, fast=True
# ))
# print("acc_ip", acc_ip(
#     data["queries"], data["docs"], data["relevancy"], 20
# ))
# print("acc_l2_fast", acc_l2(
#     data["queries"], data["docs"], data["relevancy"], 20, fast=True
# ))
# print("acc_l2", acc_l2(
#     data["queries"], data["docs"], data["relevancy"], 20
# ))

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