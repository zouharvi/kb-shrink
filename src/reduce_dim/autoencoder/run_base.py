#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_pickle, save_pickle, DEVICE
from reduce_dim.autoencoder.model import AutoencoderModel, report
import torch.nn as nn
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', default="/data/kilt-hp/dpr-c-5000.embd_cn")
parser.add_argument(
    '--data-out', default="/data/kilt-hp/tmp.embd")
parser.add_argument('--model', default=1, type=int)
parser.add_argument(
    '--bottleneck-width', default=128, type=int,
    help='Dimension of the bottleneck layer')
parser.add_argument(
    '--bottleneck-index', default=6, type=int,
    help='Position of the last encoder layer')
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
data = read_pickle(args.data)
data = {
    "queries": torch.Tensor(data["queries"]).to(DEVICE),
    "docs": torch.Tensor(data["docs"]).to(DEVICE),
    "relevancy": data["relevancy"],
}
model = AutoencoderModel(args.model, args.bottleneck_width)
print(model)

model.trainModel(data, args.epochs, bottleneck_index=-1, post_cn=args.post_cn)
model.train(False)

# encode data
with torch.no_grad():
    encoded = {
        "queries": model.encode(data["queries"], args.bottleneck_index).cpu().numpy(),
        "docs": model.encode(data["docs"], args.bottleneck_index).cpu().numpy(),
        "relevancy": data["relevancy"],
    }
report(f"Final:", encoded, data.cpu())
save_pickle(encoded, args.data_out)
