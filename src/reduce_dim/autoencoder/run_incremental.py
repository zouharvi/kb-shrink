#!/usr/bin/env python3

raise NotImplementedError(
    "Not adapted to new data orgnization (docs and queries as tuples)")

import sys
sys.path.append("src")
from misc.retrieval_utils import DEVICE
from misc.load_utils import read_pickle, save_pickle
from reduce_dim.autoencoder import Autoencoder, report
import argparse
import torch
from reduce_dim.autoencoder.model import AutoencoderModel, report

parser = argparse.ArgumentParser(
    description='Incremental autoencoder embeddings')
parser.add_argument(
    'logfile',
    help='File which to log to')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
parser.add_argument(
    '--model', default=1, type=int,
    help='Which model to use')
parser.add_argument(
    '--bottleneck-width', default=128, type=int,
    help='Dimension of the bottleneck layer')
parser.add_argument(
    '--bottleneck-index', default=5, type=int,
    help='Position of the last encoder layer')
parser.add_argument(
    '--epochs', default=1000, type=int)
parser.add_argument(
    '--seed', type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
data = read_keys_pickle(args.keys_in)
data = torch.Tensor(data).to(DEVICE)

with open(args.logfile, "a") as f:
    f.write(f"# model, width, index, acc_ip, acc_l2, avg_norm\n")

for bottleneck_width in [32, 64, 128, 256]:
    print(f"Running {bottleneck_width}")
    model = Autoencoder(args.model, bottleneck_width)
    print(model)
    model.train_routine(data, args.epochs, bottleneck_index=-1, loglevel=1)
    for bottleneck_index in range(13):
        print(f"Bottleneck index {bottleneck_index}")
        model.train(False)
        with torch.no_grad():
            encoded = model.encode(data, bottleneck_index).cpu()

        acc_ip, acc_l2, avg_norm = report(
            f"Final:", encoded, data.cpu(), level=3)
        with open(args.logfile, "a") as f:
            f.write(
                f"{args.model}, {bottleneck_width}, {bottleneck_index}, {acc_ip}, {acc_l2}, {avg_norm}\n")
