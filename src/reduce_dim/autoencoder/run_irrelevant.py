#!/usr/bin/env python3

import copy
import random
import sys
sys.path.append("src")
from misc.load_utils import read_pickle, center_data, norm_data, sub_data
from misc.retrieval_utils import rprec_a_l2, rprec_a_ip
import argparse
from reduce_dim.autoencoder.model import AutoencoderModel

parser = argparse.ArgumentParser(description='PCA performance summary')
parser.add_argument('--data')
parser.add_argument('--data-big')
parser.add_argument('--post-cn', action="store_true")
parser.add_argument('--logfile', default="computed/tmp.log")
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
data = read_pickle(args.data)
data_big = read_pickle(args.data_big)



data_train = copy.deepcopy(data)
data_train = sub_data(data_train, train=True, in_place=True)
data = sub_data(data, train=False, in_place=True)

# print({k:len(v) for k,v in data.items()})
# print({k:len(v) for k,v in data_train.items()})
# print(data["queries"][0][:5], data_train["queries"][0][:5])

logdata = []
for num_samples in [10**3, (10**3) * 3, (10**4), (10**4) * 3, 10**5, (10**5) * 3, 10**6, len(data_train["docs"]), (10**6) * 3, 10**7, (10**7) * 3]:
# for num_samples in [(10**6) * 3, 10**7, (10**7) * 3]:
    # increase test size
    if num_samples > len(data["docs"]):
        new_data = copy.deepcopy(data)
        new_data["docs"] += random.sample(
            data_big["docs"],
            num_samples - len(data["docs"])
        )

        model = AutoencoderModel(model=1, bottleneck_width=args.dim)
        model.train_routine(
            new_data, data_train,
            epochs=1,
            post_cn=args.post_cn,
            regularize=False,
            train_crop_n=None,
            train_key="d",
            skip_eval=True,
        )
        val_ip, val_l2, queries_loss, docs_loss = model.eval_routine(new_data, post_cn=args.post_cn)

        logdata.append({
            "val_ip": val_ip, "val_l2": val_l2,
            "num_samples": num_samples,
            "type": "eval_data",
        })

    # increase train size
    new_data = copy.deepcopy(data_train)
    if num_samples < len(new_data["docs"]):
        new_data["docs"] = random.sample(new_data["docs"], num_samples)
    else:
        new_data["docs"] += random.sample(
            data_big["docs"],
            num_samples - len(new_data["docs"])
        )

    model = AutoencoderModel(model=1, bottleneck_width=args.dim)
    model.train_routine(
        data, new_data,
        epochs=1,
        post_cn=args.post_cn,
        regularize=False,
        train_crop_n=None,
        train_key="d",
        skip_eval=True,
    )
    val_ip, val_l2, queries_loss, docs_loss = model.eval_routine(data, post_cn=args.post_cn)

    logdata.append({
        "val_ip": val_ip, "val_l2": val_l2,
        "num_samples": num_samples,
        "type": "train_data",
    })

    # continuously override the file
    with open(args.logfile, "w") as f:
        f.write(str(logdata))
