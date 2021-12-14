#!/usr/bin/env python3

import sys

import numpy as np
sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data
from reduce_dim.autoencoder.model import AutoencoderModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/big-hp/dpr-c.embd_cn")
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--dims', default="custom")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--center', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--train-crop-n', type=int, default=None)
parser.add_argument('--regularize', action="store_true")
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
data = read_pickle(args.data)
data = sub_data(data, train=False, in_place=True)

if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)

DIMS = process_dims(args.dims)

logdata = []
# fail first
for dim in DIMS:
    for train_key in ["dq", "d", "q"]:
        dim = int(dim)
        model = AutoencoderModel(model=args.model, bottleneck_width=dim)
        model.train_routine(
            data, args.epochs,
            post_cn=args.post_cn, regularize=args.regularize,
            train_crop_n=args.train_crop_n,
            train_key=train_key,
            skip_eval=True,
        )

        val_ip, val_l2, queries_loss, docs_loss = model.eval_routine(data, post_cn=args.post_cn)
        logdata.append({"dim": dim, "val_ip": val_ip, "val_l2": val_l2, "queries_loss": queries_loss, "docs_loss": docs_loss, "type": train_key})

        # continuously override the file
        with open(args.logfile, "w") as f:
            f.write(str(logdata))