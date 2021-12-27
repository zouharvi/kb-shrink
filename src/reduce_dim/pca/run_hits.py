#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import hits_a_ip
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data')
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
data = sub_data(data, train=False, in_place=True)


model = PCA(n_components=128).fit(data["queries"])
dataReduced = {
    "queries": model.transform(data["queries"]),
    "docs": model.transform(data["docs"])
}

dataReduced = center_data(dataReduced)
dataReduced = norm_data(dataReduced)

hits_pca = hits_a_ip(
    dataReduced["queries"],
    dataReduced["docs"],
    data["relevancy"],
    data["relevancy_articles"],
    data["docs_articles"],
    fast=True,
)

hits_orig = hits_a_ip(
    data["queries"],
    data["docs"],
    data["relevancy"],
    data["relevancy_articles"],
    data["docs_articles"],
    fast=True,
)

logdata = {
    "hits_new": hits_pca,
    "hits_old": hits_orig,
}

# continuously override the file
with open(args.logfile, "w") as f:
    f.write(str(logdata))