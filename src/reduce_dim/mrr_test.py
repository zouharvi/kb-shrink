#!/usr/bin/env python3

import sys
sys.path.append("src")

from misc.utils import read_keys_pickle, acc_l2, acc_ip
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test embedding ACC')
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

    acc_val_ip = acc_ip(data_old, data_new, 20, report=False)
    acc_val_l2 = acc_l2(data_old, data_new, 20, report=False)

    print(acc_val_ip)
    print(acc_val_l2)