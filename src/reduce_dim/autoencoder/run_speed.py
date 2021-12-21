#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.retrieval_utils import DEVICE
from misc.load_utils import process_dims, read_pickle, center_data, norm_data, sub_data
from model import AutoencoderModel
import copy
import time
import argparse
import torch

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

if args.center:
    data = center_data(data)
if args.norm:
    data = norm_data(data)
print("Because args.data_small is not provided, I'm copying the whole structure")
data_train = copy.deepcopy(data)

data = sub_data(data, train=False, in_place=True)
data_train = sub_data(data_train, train=True, in_place=True)

DIMS = process_dims(args.dims)

logdata = []
# fail first
for dim in DIMS:
    for train_key in ["dq"]:
        dim = int(dim)

        # training
        train_time = time.time()
        model = AutoencoderModel(model=args.model, bottleneck_width=dim)
        model.train_routine(
            data, data_train,
            args.epochs,
            post_cn=args.post_cn, regularize=args.regularize,
            train_crop_n=args.train_crop_n,
            train_key=train_key,
            skip_eval=True,
        )
        train_time = time.time() - train_time

        # encoding
        encode_time = time.time()
        model.train(False)
        with torch.no_grad():
            model.encode_safe_without_loss(data["queries"])
            model.encode_safe_without_loss(data["docs"])
        encode_time = time.time() - encode_time

        logdata.append({
            "dim": dim, "train_time": train_time,
            "encode_time": encode_time, "type": train_key
        })

        # continuously override the file
        with open(args.logfile, "w") as f:
            f.write(str(logdata))
