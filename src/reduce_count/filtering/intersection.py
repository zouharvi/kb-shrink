#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import intersection_ip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
args = parser.parse_args()
print("a")
data = read_pickle(args.data)
print("b")
data = sub_data(data, train=False, in_place=True)
print("c")

data["queries"] = data["queries"][:50]

if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)
print("d")

out = intersection_ip(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print("e")

for x in out:
    print(x)