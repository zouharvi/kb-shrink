#!/usr/bin/env python3

import sys
sys.path.append("src")

from misc.utils import read_keys_pickle, mrr, vec_sim_order
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore vector distribution')
    parser.add_argument(
        '--keys-in-old', default="data/eli5-dev.embd",
        help='Original embeddings')
    parser.add_argument(
        '--keys-in-new', default="data/eli5-dev-autoencoder.embd",
        help='Embeddings to compare')
    parser.add_argument(
        '--top-n', default=20, type=int,
        help='How many top elements to consider when comparing intersection')
    args = parser.parse_args()

    data_old = read_keys_pickle(args.keys_in_old)
    data_new = read_keys_pickle(args.keys_in_new)

    order_old = vec_sim_order(data_old)
    order_new = vec_sim_order(data_new)

    mrr_val = mrr(order_old, order_new, args.top_n, report=True)
