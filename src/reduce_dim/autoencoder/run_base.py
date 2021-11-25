#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.retrieval_utils import DEVICE
from misc.load_utils import read_pickle, sub_data
from reduce_dim.autoencoder.model import AutoencoderModel
import torch.nn as nn
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', default="/data/big-hp/dpr-c.embd_cn")
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--bottleneck-width', default=128, type=int)
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--train-crop-n', type=int, default=None)
parser.add_argument('--regularize', action="store_true")
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
data = read_pickle(args.data)
data = sub_data(data, train=False, in_place=True)
model = AutoencoderModel(args.model, args.bottleneck_width)
print(model)

model.trainModel(
    data, args.epochs,
    post_cn=args.post_cn, regularize=args.regularize,
    train_crop_n=args.train_crop_n,    
)
model.train(False)