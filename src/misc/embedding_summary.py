#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.utils import read_keys_pickle
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser(description='Explore embedding distribution')
parser.add_argument('keys')
args = parser.parse_args()
data = read_keys_pickle(args.keys)
data = torch.Tensor(data)

print('Data shape', data.shape)
print('Avg. L1: norm', np.average(torch.linalg.norm(data, axis=1, ord=1)))
print('Avg. L2: norm', np.average(torch.linalg.norm(data, axis=1, ord=2)))
print('Avg. Linf: norm', np.average(torch.linalg.norm(data, axis=1, ord=np.inf)))