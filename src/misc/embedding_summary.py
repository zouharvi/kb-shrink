#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Explore embedding distribution')
parser.add_argument('keys')
args = parser.parse_args()
data = read_pickle(args.keys)
print(data.keys())

def analysis(data):
    data = torch.Tensor(data)
    print('Data shape', data.shape)
    print(
        'Avg. L1 norm:  ',
        np.average(torch.linalg.norm(data, axis=1, ord=1))
    )
    print(
        'Avg. L2 norm:  ',
        np.average(torch.linalg.norm(data, axis=1, ord=2))
    )
    print(
        'Avg. Linf norm:',
        np.average(torch.linalg.norm(data, axis=1, ord=np.inf))
    )

print("QUERIES")
analysis(data["queries"])
print()
print("DOCS")
analysis(data["docs"])
