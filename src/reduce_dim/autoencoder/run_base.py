#!/usr/bin/env python3

import sys

import numpy as np
sys.path.append("src")
from misc.retrieval_utils import DEVICE
from misc.load_utils import read_pickle, sub_data
from reduce_dim.autoencoder.model import AutoencoderModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/big-hp/dpr-c.embd_cn")
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--dims', default="custom")
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--train-crop-n', type=int, default=None)
parser.add_argument('--regularize', action="store_true")
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
data = read_pickle(args.data)
data = sub_data(data, train=False, in_place=True)
if args.dims.isdigit():
    args.dims = [int(args.dims)]
elif args.dims == "custom":
    DIMS = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
elif args.dims == "linspace":
    DIMS = np.linspace(32, 768, num=768 // 32, endpoint=True)
else:
    raise Exception(f"Unknown --dims {args.dims} scheme")

logdata = []
for dim in DIMS:
    dim = int(dim)
    model = AutoencoderModel(model=args.model, bottleneck_width=dim)
    model.train_routine(
        data, args.epochs,
        post_cn=args.post_cn, regularize=args.regularize,
        train_crop_n=args.train_crop_n,
        skip_eval=True,
    )

    val_ip, val_l2, loss = model.eval_routine(data, post_cn=args.post_cn)
    logdata.append({"dim": dim, "val_ip": val_ip, "val_l2": val_l2, "loss": loss})

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))