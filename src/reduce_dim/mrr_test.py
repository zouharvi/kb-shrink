#!/usr/bin/env python3

import sys
sys.path.append("src")

from misc.utils import read_keys_pickle, mrr, vec_sim_order, l2_sim
import numpy as np
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

    order_old_ip = vec_sim_order(data_old, sim_func=np.inner)
    order_old_l2 = vec_sim_order(data_old, sim_func=l2_sim)
    order_new_ip = vec_sim_order(data_new, sim_func=np.inner)
    order_new_l2 = vec_sim_order(data_new, sim_func=l2_sim)
    mrr_val_ip = mrr(order_old_ip, order_new_ip, 20, report=False)
    mrr_val_l2 = mrr(order_old_l2, order_new_l2, 20, report=False)

    print(mrr_val_ip)
    print(mrr_val_l2)