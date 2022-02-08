#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data, zscore_data
from misc.retrieval_utils import acc_ip, acc_l2, rprec_l2, rprec_ip, rprec_a_l2, rprec_a_ip
import argparse

parser = argparse.ArgumentParser(
    description='Performance on uncompressed data'
)
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--std', action="store_true")
parser.add_argument('--all', action="store_true")
parser.add_argument('--metric', default="rprec_a")
parser.add_argument('--train', action="store_true")
parser.add_argument('--without-faiss', action="store_true")
parser.add_argument('--skip-l2', action="store_true")
parser.add_argument('--queries-n', type=int, default=None)
args = parser.parse_args()
data = read_pickle(args.data)
if args.train:
    data = sub_data(data, train=True, in_place=True)
else:
    # default is dev only
    data = sub_data(data, train=False, in_place=True)

# crop number of queries
data["queries"] = data["queries"][:args.queries_n]

if args.metric == "rprec":
    metric_l2 = rprec_l2
    metric_ip = rprec_ip
elif args.metric == "rprec_a":
    metric_l2 = rprec_a_l2
    metric_ip = rprec_a_ip
elif args.metric == "acc":
    metric_l2 = acc_l2
    metric_ip = acc_ip
else:
    raise Exception("Unknown metric")

assert not args.all or not (args.center or args.norm)

if args.all:
    data_b = data
    print(f"{args.metric}_ip:", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))
    print(f"{args.metric}_l2:", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))

    data = norm_data(data_b)
    print(f"{args.metric}_ip (norm):", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))
    print(f"{args.metric}_l2 (norm):", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))

    data = center_data(data_b)
    print(f"{args.metric}_ip (center):", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))
    print(f"{args.metric}_l2 (center):", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))

    data = norm_data(center_data(data_b))
    print(f"{args.metric}_ip (center, norm):", "{:.4f}".format(metric_ip(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))
    print(f"{args.metric}_l2 (center, norm):", "{:.4f}".format(metric_l2(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True
    )))
else:
    if args.center:
        data = center_data(data)
    if args.std:
        data = zscore_data(data, center=False)
    if args.norm:
        data = norm_data(data)

    # RPrec
    print(f"{args.metric}_ip (fast)", metric_ip(
        data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True,
    ))
    if args.without_faiss:
        print(f"{args.metric}_ip", metric_ip(
            data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=False,
        ))

    if not args.skip_l2:
        print(f"{args.metric}_l2 (fast)", metric_l2(
            data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=True,
        ))
        if args.without_faiss:
            print(f"{args.metric}_l2", metric_l2(
                data["queries"], data["docs"], data["relevancy"], data["relevancy_articles"], data["docs_articles"], fast=False,
            ))
