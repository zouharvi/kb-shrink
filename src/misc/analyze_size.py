#!/usr/bin/env python3

import pickle
import argparse
import sys
import torch
import random
import numpy as np
from utils import read_keys_pickle, load_dataset
from pympler.asizeof import asizeof

parser = argparse.ArgumentParser(description='Explore vector distribution')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
parser.add_argument(
    '--dataset', default="data/eli5-dev.jsonl",
    help='KILT (sub)dataset with JSON lines')
args = parser.parse_args()

keys = read_keys_pickle(args.keys_in)
keys_size = asizeof(keys)

dataset_all = load_dataset(args.dataset, keep="all")
dataset_size = asizeof(dataset_all)

dataset_prompts = load_dataset(args.dataset, keep="inputs")
prompts_size = asizeof(dataset_prompts)

dataset_answers = load_dataset(args.dataset, keep="answers")
answers_size = asizeof(dataset_answers)

print(f"Whole dataset size:  {dataset_size/1024/1024:>5.1f}MB")
print(f"Prompts size:        {prompts_size/1024:>5.1f}KB", f"{prompts_size/dataset_size*100:>4.1f}%")
print(f"Values size:         {answers_size/1024/1024:>5.1f}MB", f"{answers_size/dataset_size*100:>4.1f}%")
print(f"Keys size:           {keys_size/1024/1024:>5.1f}MB", f"{keys_size/answers_size:>4.1f}x values size")
print(f"Keys size (calc):    {8*keys[0].shape[0]*len(keys)/1024/1024:>5.1f}MB")
print(f"One key size:        {asizeof(keys[0].tolist())/1024:>5.1f}KB")
print(f"One key size (calc): {8*keys[0].shape[0]/1024:>5.1f}KB")
print(f"Number of entries:   {len(dataset_all):>7}")
