#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_keys_pickle, save_keys_pickle, DEVICE, mrr_ip_fast, mrr_l2_fast
import argparse
import numpy as np
import torch
import torch.nn as nn

def report(prefix, encoded, data, level):
    # V^2 similarity computations is computationally expensive, skip if not necessary
    if level == 3:
        mrr_val_ip = mrr_ip_fast(data, encoded, 20, report=False)
        mrr_val_l2 = mrr_l2_fast(data, encoded, 20, report=False)
        avg_norm = np.average(torch.linalg.norm(encoded, axis=1))
        print(f'{prefix} mrr_ip: {mrr_val_ip:.3f}, mrr_l2: {mrr_val_l2:.3f}, norm: {avg_norm:.2f}')
        return mrr_val_ip, mrr_val_l2, avg_norm
    elif level == 2:
        mrr_val_ip = mrr_ip_fast(data, encoded, 20, report=False)
        print(f'{prefix} mrr_ip: {mrr_val_ip:.3f}')
        return mrr_val_ip
    elif level == 1:
        print(f'{prefix}')

class Autoencoder(nn.Module):
    def __init__(self, model, dimension=64, batchSize=128, learningRate=0.0001):
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
        self.similarity = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x1, x2):
        y1 = self.projection(x1)
        y2 = self.projection(x2)
        return self.similarity(y1, y2)

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
                sample_sim = self.criterion(sample1, sample2)
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
    data = read_keys_pickle(args.keys_in)[:5000]
    data = torch.Tensor(data).to(DEVICE)
    model = Autoencoder(args.model, args.dimension)
    print(model)
    model.trainModel(data, args.epochs, loglevel=args.loglevel)
    model.train(False)
    with torch.no_grad():
        encoded = model.encode(data).cpu()
    report(f"Final:", encoded, data.cpu(), level=3)
    save_keys_pickle(encoded, args.keys_out)