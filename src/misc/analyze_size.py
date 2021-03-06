#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, read_json
import argparse
from pympler.asizeof import asizeof
import numpy as np

parser = argparse.ArgumentParser(description='Explore vector distribution')
parser.add_argument('--embd', default="/data/big-hp/dpr-c.embd")
parser.add_argument('--dataset', default="/data/big-hp/full.pkl")
args = parser.parse_args()

data = read_pickle(args.embd)
sizee_all = asizeof(data)
sizee_queries = asizeof(data["queries"])
sizee_docs = asizeof(data["docs"])

data_raw = read_pickle(args.dataset)
sizer_all = asizeof(data_raw)
sizer_queries = asizeof(data_raw["queries"])
sizer_docs = asizeof(data_raw["docs"])

unitType = str(data["queries"][0].dtype)
if unitType == "float64":
    unitSize = 8
elif unitType == "float32":
    unitSize = 4
elif unitType == "float16":
    unitSize = 2

print(f"Whole raw size:      {sizer_all/1024/1024:>5.1f}MB")
print(f"Prompts size (raw):  {sizer_queries/1024/1024:>5.1f}MB", f"{sizer_queries/sizer_all*100:>4.1f}%")
print(f"Docs size (raw):     {sizer_docs/1024/1024:>5.1f}MB", f"{sizer_docs/sizer_all*100:>4.1f}%")
print()
print(f"Whole embd size:     {sizee_all/1024/1024:>5.1f}MB")
print(f"Prompts size (embd): {sizee_queries/1024/1024:>5.1f}MB", f"{sizee_queries/sizee_all*100:>4.1f}%")
print(f"Docs size (embd):    {sizee_docs/1024/1024:>5.1f}MB", f"{sizee_docs/sizee_all*100:>4.1f}%")
print()
print(f"Query shape:         {data['queries'][0].shape}")
print(f"Query element type:  {unitType}")
print()
print(f"Number of queries:   {len(data['queries']):>7}")
print(f"Number of docs:      {len(data['docs']):>7}")
print()
print(f'Average number of spans per question: {np.average([len(x) for x in data["relevancy"]]):.2f}')
print(f'Average number of spans per document: {np.average([len(x) for x in data["data_docs_articles"]]):.2f}')
print()

data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

print(f"Average of query embedding 1-norm:",
    "{:.3f}".format(np.average(np.linalg.norm(data["queries"], axis=1, ord=1))),
    "std:",
    "{:.3f}".format(np.std(np.linalg.norm(data["queries"], axis=1, ord=1)))
)
print(f"Average of doc embedding 1-norm:  ",
    "{:.3f}".format(np.average(np.linalg.norm(data["docs"], axis=1, ord=1))),
    "std:",
    "{:.3f}".format(np.std(np.linalg.norm(data["docs"], axis=1, ord=1)))
)
print()
print(f"Average of query embedding 2-norm:",
    "{:.3f}".format(np.average(np.linalg.norm(data["queries"], axis=1, ord=2))),
    "std:",
    "{:.3f}".format(np.std(np.linalg.norm(data["queries"], axis=1, ord=2)))
)
print(f"Average of doc embedding 2-norm:  ",
    "{:.3f}".format(np.average(np.linalg.norm(data["docs"], axis=1, ord=2))),
    "std:",
    "{:.3f}".format(np.std(np.linalg.norm(data["docs"], axis=1, ord=2)))
)