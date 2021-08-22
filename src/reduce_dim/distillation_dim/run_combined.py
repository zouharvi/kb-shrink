#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import read_pickle, save_pickle
from misc.retrieval_utils import DEVICE
import argparse
import torch

raise NotImplementedError()

from reduce_dim.distillation_dim.model_combined import SimDistilModelCombined, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', default="/data/kilt/hotpot-dpr-c-5000.embd_norm")
    parser.add_argument('--data-out')
    parser.add_argument('--dimension', default=128, type=int)
    parser.add_argument('--model', default=2, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--batch-size', default=2500, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--data-organization', default="dd")
    parser.add_argument('--post-cn', action="store_true")
    parser.add_argument('--not-merge', action="store_true")
    parser.add_argument('--similarity-model', default="l2")
    parser.add_argument('--similarity-gold', default="l2")
    parser.add_argument('--seed', type=int, default=0)
    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    data = read_pickle(args.data)
    data = {
        "queries": torch.Tensor(data["queries"]).to(DEVICE),
        "docs": torch.Tensor(data["docs"]).to(DEVICE),
        "relevancy": data["relevancy"], 
    }
    model = SimDistilModelCombined(
        args.model, args.dimension,
        batchSize=args.batch_size,
        learningRate=args.learning_rate,
        dataOrganization=args.data_organization,
        merge=not args.not_merge,
        similarityModel=args.similarity_model,
        similarityGold=args.similarity_gold,
    )
    print(model)
    model.trainModel(data, args.epochs, args.post_cn)
    model.train(False)

    # encode data
    with torch.no_grad():
        encoded = {
            "queries": model.encode1(data["queries"]).cpu().numpy(),
            "docs": model.encode2(data["docs"]).cpu().numpy(), 
            "relevancy": data["relevancy"], 
        } 
    report(f"Final:", encoded, data.cpu())
    save_pickle(encoded, args.data_out)