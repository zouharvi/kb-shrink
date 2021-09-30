#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import argparse
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c-5000.embd_cn")
parser.add_argument('--center', action="store_true")
parser.add_argument('--std', action="store_true")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)

print("Preparing model")
model_d = preprocessing.StandardScaler(
    with_std=args.std, with_mean=args.center
).fit(data["docs"])
model_q = preprocessing.StandardScaler(
    with_std=args.std, with_mean=args.center
).fit(data["queries"])

dataNew = {
    "docs": model_d.transform(data["docs"]),
    "queries": model_q.transform(data["queries"]),
}
print(data["docs"][0][:5])
print(dataNew["docs"][0][:5])

val_ip_pca = rprec_a_ip(
    dataNew["queries"], dataNew["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
val_l2_pca = rprec_a_l2(
    dataNew["queries"], dataNew["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (zscores)")

val_ip_pca = rprec_a_ip(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
val_l2_pca = rprec_a_l2(
    data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
)
print(f"ip: {val_ip_pca:.4f}, l2: {val_l2_pca:.4f} (orig)")
