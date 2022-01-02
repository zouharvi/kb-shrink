#!/usr/bin/env python3

import copy
import sys

sys.path.append("src")
from misc.load_utils import process_dims, read_pickle, sub_data, IdentityScaler, NormScaler, CenterScaler
from reduce_dim.autoencoder.model import AutoencoderModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/hp/dpr-c.embd_cn")
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

data_train = copy.deepcopy(data)
data = sub_data(data, train=False, in_place=True)
data_train = sub_data(data_train, train=True, in_place=True)
data_orig = copy.deepcopy(data)

# see pca/run_base.py for new preprocessing scheme
if args.center:
    scaler_center = CenterScaler()
    data = scaler_center.transform(data)
    data_train = CenterScaler().transform(data_train)
else:
    scaler_center = IdentityScaler()

if args.norm:
    scaler_norm = NormScaler()
    data = scaler_norm.transform(data)
    data_train = NormScaler().transform(data_train)
else:
    scaler_norm = IdentityScaler()

DIMS = process_dims(args.dims)

logdata = []
# fail first
for dim in DIMS:
    for train_key in ["dq", "d", "q"]:
        dim = int(dim)
        model = AutoencoderModel(model=args.model, bottleneck_width=dim)
        model.train_routine(
            data, data_train,
            args.epochs,
            post_cn=args.post_cn, regularize=args.regularize,
            train_crop_n=args.train_crop_n,
            train_key=train_key,
            skip_eval=True,
        )

        val_ip, val_l2, loss_q, loss_d = model.eval_routine(
            data=data, data_orig=data_orig,
            scaler_center=scaler_center, scaler_norm=scaler_norm,
            post_cn=args.post_cn
        )
        logdata.append({
            "dim": dim, "val_ip": val_ip, "val_l2": val_l2,
            "loss_q": loss_q, "loss_d": loss_d, "type": train_key
        })

        # continuously override the file
        with open(args.logfile, "w") as f:
            f.write(str(logdata))
