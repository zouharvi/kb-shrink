#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_pickle, save_pickle, DEVICE
import argparse
import torch
from distillation_dim.model import ProjectionModel, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', default="/data/kilt/hotpot-dpr-c-5000.embd_norm")
    parser.add_argument('--data-out')
    parser.add_argument(
        '--dimension', default=64, type=int)
    parser.add_argument(
        '--model', default=2, type=int)
    parser.add_argument(
        '--epochs', default=10000, type=int)
    parser.add_argument(
        '--log-level', default=1, type=int)
    parser.add_argument(
        '--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = read_pickle(args.data)
    data = {
            "queries": torch.Tensor(data["queries"]).to(DEVICE),
            "docs": torch.Tensor(data["docs"]).to(DEVICE),
            "relevancy": data["relevancy"], 
        } 
    model = ProjectionModel(args.model, args.dimension)
    print(model)
    model.trainModel(data, args.epochs, loglevel=args.log_level)
    model.train(False)

    # encode data
    with torch.no_grad():
        encoded = {
            "queries": model.encode1(data["queries"]).cpu().numpy(),
            "docs": model.encode2(data["queries"]).cpu().numpy(), 
            "relevancy": data["relevancy"], 
        } 
    report(f"Final:", encoded, data.cpu(), level=2)
    save_pickle(encoded, args.data_out)