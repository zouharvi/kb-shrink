#!/usr/bin/env python3

import sys
sys.path.append("src")

from misc.utils import read_keys_pickle, save_keys_pickle, DEVICE
import argparse
import numpy as np

def sim_order(data):
    order = [
        [np.inner(vec1, vec2) for vec2 in data]
        for vec1 in data
    ]
    return [
        sorted(
            range(len(data)),
            key=lambda x: sims[x],
            reverse=True
        ) for sims in order
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore vector distribution')
    parser.add_argument(
        '--keys-in-old', default="data/eli5-dev.embd",
        help='Original embeddings')
    parser.add_argument(
        '--keys-in-new', default="data/eli5-dev-autoencoder.embd",
        help='Embeddings to compare')
    parser.add_argument(
        '--top-n', default=100, type=int,
        help='How many top elements to consider when comparing intersection')
    args = parser.parse_args()

    data_old = read_keys_pickle(args.keys_in_old)
    data_new = read_keys_pickle(args.keys_in_new)

    order_old = sim_order(data_old)
    order_new = sim_order(data_new)
    
    avg_overlap = np.average([len(set(x[:args.top_n]) & set(y[:args.top_n])) for x,y in zip(order_old, order_new)])
    print(f"Average overlap (top {args.top_n}) is {args.top_n} ({avg_overlap/args.top_n*100:.2f}%)")
