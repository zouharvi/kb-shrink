#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.utils import read_keys_pickle, load_dataset
import argparse
from pympler.asizeof import asizeof

parser = argparse.ArgumentParser(description='Explore vector distribution')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
parser.add_argument(
    '--dataset', default="data/eli5-dev.jsonl",
    help='KILT (sub)dataset with JSON lines')
parser.add_argument(
    '--dataset-type', default="eli5",
    help='eli5 or hotpot')
args = parser.parse_args()

keys = read_keys_pickle(args.keys_in)
keys_size = asizeof(keys)

dataset_all = load_dataset(args.dataset, keep="all")
dataset_size = asizeof(dataset_all)

dataset_prompts = load_dataset(args.dataset, keep="inputs")
prompts_size = asizeof(dataset_prompts)

if args.dataset_type == "eli5":
    dataset_answers = load_dataset(args.dataset, keep="answers")
    answers_size = asizeof(dataset_answers)
elif args.dataset_type == "hotpot":
    # drop provenance
    dataset_answers = load_dataset(args.dataset, keep="answers")
    answers_size = asizeof(dataset_answers)
else:
    raise Exception("Unknown dataset type")

unitType = str(keys[0].dtype)
if unitType == "float64":
    unitSize = 8
elif unitType == "float32":
    unitSize = 4
elif unitType == "float16":
    unitSize = 2

print(f"Whole dataset size:  {dataset_size/1024/1024:>5.1f}MB")
print(f"Prompts size:        {prompts_size/1024/1024:>5.1f}MB", f"{prompts_size/dataset_size*100:>4.1f}%")
print(f"Values size:         {answers_size/1024/1024:>5.1f}MB", f"{answers_size/dataset_size*100:>4.1f}%")
print(f"Keys size (comp):    {keys_size/1024/1024:>5.1f}MB", f"{keys_size/answers_size:>4.1f}x values size")
print(f"Keys size (calc):    {unitSize*keys[0].shape[0]*len(keys)/1024/1024:>5.1f}MB")
print(f"One key size (comp): {asizeof(keys[0].tolist())/1024:>5.1f}KB")
print(f"One key size (calc): {unitSize*keys[0].shape[0]/1024:>5.1f}KB")
print(f"Key shape:           {keys[0].shape}")
print(f"Key element type:    {unitType}")
print(f"Number of entries:   {len(dataset_all):>7}")
