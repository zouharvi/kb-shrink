#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data
from misc.retrieval_utils import rprec_l2, rprec_ip, DEVICE
from reduce_dim.autoencoder.model import AutoencoderModel
import argparse
import timeit
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="/data/kilt-hp/dpr-c.embd_cn")
parser.add_argument('--logfile', default="computed/asingle_time.log")
# parser.add_argument('--thresholds', nargs='+')
parser.add_argument('--thresholds-epochs', action="store_true")
parser.add_argument('--step', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=int, default=1)
args = parser.parse_args()
data = read_pickle(args.data)


logdata = []
EPOCHS = {
    1000: 90,
    5000: 75,
    10000: 50,
    15000: 35,
    20000: 25,
    40000: 30,
    60000: 25,
    80000: 15,
    100000: 15,
    len(data['docs']): 15
}

# override if argument passed
if args.thresholds_epochs:
    thresholds = list(EPOCHS.keys())
else:
    thresholds = range(1000, len(data['docs'])+args.step-1, args.step)
    print(f"Making {(len(data['docs'])-1000)//args.step} steps from {1000} (base) to {len(data['docs'])} (total doc count)")


data = {
    "queries": torch.Tensor(data["queries"]).to(DEVICE),
    "docs": torch.Tensor(data["docs"]).to(DEVICE),
    "relevancy": data["relevancy"],
}

for threshold in thresholds:
    threshold = min(threshold, len(data['docs']))

    model = AutoencoderModel(
        model=args.model,
        bottleneck_width=128,
    )

    train_time = timeit.timeit(
        lambda: model.trainModel(
            data, EPOCHS[threshold],
            post_cn=True, regularize=True,
            skip_eval=True, train_crop_n=threshold
        ),
        number=1,
    )

    model.train(False)
    with torch.no_grad():
        dataReduced = {
            "queries": model.encode(data["queries"]).cpu(),
            "docs": model.encode(data["docs"]).cpu(),
        }
    
    dataReduced = center_data(dataReduced)
    dataReduced = norm_data(dataReduced)

    # we don't need L2 because the result on normalized data is identical to IP
    val_ip_pca = rprec_ip(
        dataReduced["queries"], dataReduced["docs"], data["relevancy"], fast=True
    )
    print(f"threshold: {threshold}, ip: {val_ip_pca:.4f}, train_time: {train_time:.2f}")
    logdata.append({"threshold": threshold, "val_ip": val_ip_pca, "type": "pca", "train_time": train_time})

    # override dump
    with open(args.logfile, "w") as f:
        f.write(str(logdata))