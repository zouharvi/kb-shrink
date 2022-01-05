#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import intersection_ip, acc_ip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c-pruned.embd")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
args = parser.parse_args()
data = read_pickle(args.data)
data = sub_data(data, train=False, in_place=True)
del data["relevancy_articles"]
del data["docs_articles"]

data["queries"] = data["queries"][4:5]
data["relevancy"] = data["relevancy"][4:5]

if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)

print(f"Running acc on {len(data['docs'])} docs")
val_ip = acc_ip(
    data["queries"], data["docs"], data["relevancy"], n=20
)
print(f"Acc-20: {val_ip:.4f}")

print(f"Running retrieval on {len(data['docs'])} docs")
intersection = intersection_ip(
    data["queries"], data["docs"], data["relevancy"], n=20
)

print("Computing which documents to prune")
to_prune = []
for docs_relevant, docs_all in zip(intersection, data["relevancy"]):
    # add docs which were previously relevant but did not help in retrieval
    to_prune += set(docs_all) - docs_relevant
    print(docs_relevant)
print(to_prune)
print("To-prune:", len(to_prune))

print("Pruning documents")
offset = 0
offset_relevancy = []
for doc in to_prune:
    offset_relevancy.append(doc)
    doc -= offset
    data["docs"].pop(doc)
    offset += 1
print("Docs:", len(data["docs"]))

print("Offseting relevancy")
# offset_relevancy.sort()
# prev_offset = -1
# offset_relevancy_neat = {-1: 0}
# for offset in offset_relevancy:
#     offset_relevancy_neat[offset] = offset_relevancy_neat[prev_offset] + 1
#     prev_offset = offset

# # remove phony target
# offset_relevancy_neat.pop(-1)
# offset_relevancy = sorted(offset_relevancy_neat.items(),
#                           key=lambda x: x[0], reverse=True)


# def update_doc_offset(doc):
#     offset = 0
#     for doc_offset, offset_val in offset_relevancy:
#         if doc_offset <= doc:
#             offset = offset_val
#             break
#     return doc - offset


# data["relevancy"] = [
#     [update_doc_offset(x) for x in relevancy] for relevancy in data["relevancy"]
# ]

print(data["relevancy"])
# TODO: there's a faster way to do this with just a single pass
for offset in offset_relevancy:
    data["relevancy"] = [[x-1 if x > offset else x for x in relevancy] for relevancy in data["relevancy"]]
print(data["relevancy"])

print(f"Running acc on {len(data['docs'])} docs")
val_ip = acc_ip(
    data["queries"], data["docs"], data["relevancy"], n=20
)
print(f"Acc-20: {val_ip:.4f}")
