#!/usr/bin/env python3

import pickle
import argparse
import random

parser = argparse.ArgumentParser(
    description='Reduce KB key count')
parser.add_argument(
    '--prob', type=float, default=0.5,
    help='Probability by which to DISCARD every key')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
parser.add_argument(
    '--keys-out', default="data/eli5-dev-random.embd",
    help='Output keys')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

with open(args.keys_in, "rb") as fread, open(args.keys_out, "wb") as fwrite:
    reader = pickle.Unpickler(fread)
    writer = pickle.Pickler(fwrite)
    while True:
        try:
            data = reader.load()
            if random.uniform(0, 1) >= args.prob:
                writer.dump(data)
        except EOFError:
            break
