#!/usr/bin/env python3

import sys
sys.path.append("src")

from misc.utils import read_keys_pickle, save_keys_pickle, DEVICE
import argparse
from pympler.asizeof import asizeof
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, model, batchSize=128, learningRate=0.001):
        super().__init__()

        if model == 1:
            # Encoder Network
            self.encoder = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 64)
            )
            # Decoder Network
            self.decoder = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(True),
                nn.Linear(256, 512),
                nn.ReLU(True),
                nn.Linear(512, 768),
                nn.Tanh(),
            )
            # TODO: remove tanh?
        else:
            raise Exception("Unknown model specified")

        self.batchSize = batchSize
        self.learningRate = learningRate

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )
        self.criterion = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def trainModel(self, data, epochs):
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.batchSize, shuffle=True
        )

        for epoch in range(epochs):
            for sample in self.dataLoader:
                # Predictions
                output = self(sample)
                # Calculate Loss
                loss = self.criterion(output, sample)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'epoch [{epoch+1}/{epochs}], loss:{loss.data:.7f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore vector distribution')
    parser.add_argument(
        '--keys-in', default="data/eli5-dev.embd",
        help='Input keys')
    parser.add_argument(
        '--keys-out', default="data/eli5-dev-autoencoder.embd",
        help='Encoded keys')
    parser.add_argument(
        '--model', default=1, type=int,
        help='Which model to use (1 - big)')
    parser.add_argument(
        '--epochs', default=1000, type=int)
    parser.add_argument(
        '--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = read_keys_pickle(args.keys_in)
    data = torch.Tensor(data).to(DEVICE)
    model = Autoencoder(args.model)
    model.trainModel(data, args.epochs)
    model.train(False)
    with torch.no_grad():
        encoded = model.encode(data).cpu()
    save_keys_pickle(encoded, args.keys_out)
