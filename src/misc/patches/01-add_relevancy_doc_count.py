#!/usr/bin/env python3

"""
This patch is necessary so that r-precision may use correct information
regarding the articles.
"""

import sys; sys.path.append("src")
from misc.load_utils import save_pickle, read_pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-1', default="/data/hp/full.pkl")
parser.add_argument('--data-2', default="/data/hp/full.embd")
parser.add_argument('--data-out', default="/data/hp/full_fixed.embd")

args = parser.parse_args()
data1 = read_pickle(args.data_1)
data2 = read_pickle(args.data_2)

if "relevancy" not in data2:
    raise Exception("Second data does not have relevancy entry")
if "relevancy_articles" not in data2:
    raise Exception("Second data does not have relevancy_articles entry")
if "docs_articles" not in data2:
    raise Exception("Second data does not have docs_articles entry")
if len(data1["queries"]) != len(data2["queries"]):
    raise Exception("Data lengths (queries) are not matching")
if len(data1["docs"]) != len(data2["docs"]):
    raise Exception("Data lengths (docs) are not matching")

print("query type1:", type(data1["queries"][0]))
print("docs type1:", type(data1["docs"][0]))
print("query type2:", type(data2["queries"][0]))
print("docs type2:", type(data2["docs"][0]))
print("relevancy type2:", type(data2["relevancy"][0]))
print("relevancy_articles type2:", type(data2["relevancy_articles"][0]))
print("relevancy_articles item type2:", type(list(data2["relevancy_articles"][0])[0]))
print("docs_articles type2:", type(data2["docs_articles"][0]))

# print("Changing article ids to ints")
# data2["docs_articles"] = [int(x) for x in data2["docs_articles"]]

print("Updating data1")
data1["relevancy"] = data2["relevancy"]
data1["relevancy_articles"] = data2["relevancy_articles"]
data1["docs_articles"] = data2["docs_articles"]

print("Examples")
print("data_query[0]:")
print(data1["queries"][0])
print("\ndata_docs[0]:")
print(data1["docs"][0])
print("\ndata_relevancy[0]:")
print(data1["relevancy"][0])
print("\ndata_relevancy_articles[0]:")
print(data1["relevancy_articles"][0])
print("\ndata_docs_articles[0]:")
print(data1["docs_articles"][0])

print("Saving")
save_pickle(args.data_out, data1)