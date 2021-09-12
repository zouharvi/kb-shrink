#!/usr/bin/env python3

"""
Change the data type of relevancy_articles item
"""

import sys; sys.path.append("src")
from misc.load_utils import save_pickle, read_pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/big-hp/dpr-c.pkl")
parser.add_argument('--data-out', default="/data/big-hp/dpr-c_fixed.embd")
args = parser.parse_args()
data = read_pickle(args.data)

print("relevancy_articles type:", type(data["relevancy_articles"][0]))
print("relevancy_articles item type:", type(list(data["relevancy_articles"][0])[0]))
print("docs_articles type:", type(data["docs_articles"][0]))

print("Updating")
data["relevancy_articles"] = [{int(art) for art in relevancy} for relevancy in data["relevancy_articles"]]
data["docs_articles"] = [int(art) for art in data["docs_articles"]]

print("relevancy_articles type:", type(data["relevancy_articles"][0]))
print("relevancy_articles item type:", type(list(data["relevancy_articles"][0])[0]))
print("docs_articles type:", type(data["docs_articles"][0]))

print("Saving")
save_pickle(args.data_out, data)