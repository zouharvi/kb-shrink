#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import read_pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Basic embedding file statistics')
parser.add_argument('--data', default="/data/hp/dpr-c.embd")
args = parser.parse_args()

data = read_pickle(args.data)

print(f"Number of queries:   {len(data['queries']):>7}")
print(f"Number of docs:      {len(data['docs']):>7}")
print(f"Query shape:         {data['queries'][0].shape}")
print(f"Query element type:  {str(data['queries'][0].dtype)}")
print()
print()
print("Boundaries:", data["boundaries"])
print()
print(
    f'Average number of spans per question: {np.average([len(x) for x in data["relevancy"]]):.2f}'
)
print(
    f'Average number of spans per document: {np.average([len(x) for x in data["relevancy_articles"]]):.2f}'
)
print()

data["queries"] = np.array(data["queries"])
data["docs"] = np.array(data["docs"])

print(f"Average of query embedding 1-norm:",
      "{:.3f}".format(np.average(np.linalg.norm(
          data["queries"], axis=1, ord=1))),
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
      "{:.3f}".format(np.average(np.linalg.norm(
          data["queries"], axis=1, ord=2))),
      "std:",
      "{:.3f}".format(np.std(np.linalg.norm(data["queries"], axis=1, ord=2)))
      )
print(f"Average of doc embedding 2-norm:  ",
      "{:.3f}".format(np.average(np.linalg.norm(data["docs"], axis=1, ord=2))),
      "std:",
      "{:.3f}".format(np.std(np.linalg.norm(data["docs"], axis=1, ord=2)))
      )
