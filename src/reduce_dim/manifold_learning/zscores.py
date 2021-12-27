#!/usr/bin/env python3

import sys

from numpy.core.defchararray import center
sys.path.append("src")
from misc.load_utils import read_pickle, zscore_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--std', action="store_true")
args = parser.parse_args()
data = read_pickle(args.data)

print(data["docs"][0][:5])

print("Preparing model")
data = zscore_data(data, center=True)
print(data["docs"][0][:5])

val_ip_pca = rprec_a_ip(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
val_l2_pca = rprec_a_l2(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (zscores)")

val_ip_pca = rprec_a_ip(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
val_l2_pca = rprec_a_l2(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (orig)")
