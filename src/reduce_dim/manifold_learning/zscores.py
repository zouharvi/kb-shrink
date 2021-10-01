#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import argparse
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/big-hp/dpr-c.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--std', action="store_true")
args = parser.parse_args()
data = read_pickle(args.data)

print("Preparing model")
data["docs"] = preprocessing.StandardScaler(
    with_std=args.std, with_mean=args.center
).fit_transform(data["docs"])
data["queries"] = preprocessing.StandardScaler(
    with_std=args.std, with_mean=args.center
).fit_transform(data["queries"])

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
