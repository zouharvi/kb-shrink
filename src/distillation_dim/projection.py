#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_keys_pickle, save_keys_pickle, DEVICE, acc_ip, acc_l2
import argparse
import numpy as np
import torch
import torch.nn as nn

def report(prefix, encoded, data, level):
    # V^2 similarity computations is computationally expensive, skip if not necessary
    if level == 3:
        acc_val_ip = acc_ip(data, encoded, 20, report=False)
        acc_val_l2 = acc_l2(data, encoded, 20, report=False)
        avg_norm = np.average(torch.linalg.norm(encoded, axis=1))
        print(f'{prefix} acc_ip: {acc_val_ip:.3f}, acc_l2: {acc_val_l2:.3f}, norm: {avg_norm:.2f}')
        return acc_val_ip, acc_val_l2, avg_norm
    elif level == 2:
        acc_val_ip = acc_ip(data, encoded, 20, report=False)
        print(f'{prefix} acc_ip: {acc_val_ip:.3f}')
        return acc_val_ip
    elif level == 1:
        print(f'{prefix}')
        
# TODO:  try lower learning rate
class Autoencoder(nn.Module):
    def __init__(self, model, dimension=64, batchSize=5000, learningRate=0.0001):
        super().__init__()

        if model == 1:
            self.projection = nn.Linear(768, dimension)
        elif model == 2:
            self.projection = nn.Sequential(
                nn.Linear(768, 512),
                nn.Tanh(),
                nn.Linear(512, dimension),
            )
        else:
            raise Exception("Unknown model specified")

        self.batchSize = batchSize
        self.learningRate = learningRate

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )
        self.similarity = nn.PairwiseDistance(p=2)
        self.criterion = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x1, x2):
        y1 = self.projection(x1)
        y2 = self.projection(x2)
        out = self.similarity(y1, y2)
        return out

    def encode(self, x):
        return self.projection(x)

    def trainModel(self, data, epochs, loglevel):
        self.dataLoader1 = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.batchSize, shuffle=True
        )
        self.dataLoader2 = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.batchSize, shuffle=True
        )

        for epoch in range(epochs):
            self.train(True)
            for sample1, sample2 in zip(self.dataLoader1, self.dataLoader2):
                # Predictions
                output = self(sample1, sample2)
                sample_sim = self.similarity(sample1, sample2)
                # Calculate Loss
                loss = self.criterion(output, sample_sim)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = self.encode(data).cpu()

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.9f},",
                    encoded, data.cpu(), level=loglevel
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder dimension reduction')
    parser.add_argument(
        '--keys-in', default="data/eli5-dev.embd",
        help='Input keys')
    parser.add_argument(
        '--keys-out', default="data/eli5-dev-distiller.embd",
        help='Encoded keys')
    parser.add_argument(
        '--dimension', default=64, type=int)
    parser.add_argument(
        '--model', default=1, type=int)
    parser.add_argument(
        '--epochs', default=10000, type=int)
    parser.add_argument(
        '--loglevel', default=1, type=int,
        help='Level at which to report')
    parser.add_argument(
        '--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = read_keys_pickle(args.keys_in)
    data = torch.Tensor(data).to(DEVICE)
    model = Autoencoder(args.model, args.dimension)
    print(model)
    model.trainModel(data, args.epochs, loglevel=args.loglevel)
    model.train(False)
    with torch.no_grad():
        encoded = model.encode(data).cpu()
    report(f"Final:", encoded, data.cpu(), level=3)
    save_keys_pickle(encoded, args.keys_out)