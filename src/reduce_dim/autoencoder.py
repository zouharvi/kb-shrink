#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import l2_sim

from misc.utils import mrr, read_keys_pickle, save_keys_pickle, DEVICE, vec_sim_order
import argparse
import numpy as np
import torch
import torch.nn as nn

order_old_ip = None
order_old_l2 = None


def report(prefix, encoded, data, level, order_old_ip=None, order_old_l2=None):
    if level >= 1:
        if order_old_ip is None:
            order_old_ip = vec_sim_order(data, sim_func=np.inner)

    if level >= 2:
        if order_old_l2 is None:
            order_old_l2 = vec_sim_order(data, sim_func=l2_sim)

    # V^2 similarity computations is computationally expensive, skip if not necessary
    if level == 2:
        order_new = vec_sim_order(encoded, sim_func=np.inner)
        mrr_val_ip = mrr(order_old_ip, order_new, 20, report=False)
        order_new = vec_sim_order(encoded, sim_func=l2_sim)
        mrr_val_l2 = mrr(order_old_l2, order_new, 20, report=False)
        print(f'{prefix} mrr_ip: {mrr_val_ip:.3f}, mrr_l2: {mrr_val_l2:.3f}')
    elif level == 1:
        order_new = vec_sim_order(encoded)
        mrr_val_ip = mrr(order_old_ip, order_new, 20, report=False)
        print(f'{prefix} mrr_ip: {mrr_val_ip:.3f}')
    elif level == 0:
        print(f'{prefix}')


class Autoencoder(nn.Module):
    def __init__(self, model, bottleneck_width, bottleneck_index, batchSize=128, learningRate=0.001):
        super().__init__()

        if model == 1:
            # Encoder Network
            layers = [
                nn.Linear(768, 512),               # 1
                nn.Tanh(),                         # 2
                nn.Linear(512, 256),               # 3
                nn.Tanh(),                         # 4
                nn.Linear(256, bottleneck_width),  # 5
                nn.Tanh(),                         # 6
                nn.Linear(bottleneck_width, 256),  # 7
                nn.Tanh(),                         # 8
                nn.Linear(256, 512),               # 9
                nn.Tanh(),                         # 10
                nn.Linear(512, 768),               # 11
                nn.Tanh(),                         # 12
            ]
            # One may be tempted to remove the Tanh, but it worsens the performance
            self.encoder = nn.Sequential(*layers[:bottleneck_index])
            self.decoder = nn.Sequential(*layers[bottleneck_index:])
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
            self.train(True)
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
                self.train(False)
                with torch.no_grad():
                    encoded = model.encode(data).cpu()

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.7f},",
                    encoded, data.cpu(), level=args.level
                )


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
        '--bottleneck-width', default=256, type=int,
        help='Dimension of the bottleneck layer')
    parser.add_argument(
        '--bottleneck-index', default=5, type=int,
        help='Position of the last encoder layer')
    parser.add_argument(
        '--epochs', default=1000, type=int)
    parser.add_argument(
        '--level', default=0, type=int,
        help='Level at which to report')
    parser.add_argument(
        '--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = read_keys_pickle(args.keys_in)
    data = torch.Tensor(data).to(DEVICE)
    model = Autoencoder(args.model, args.bottleneck_width, args.bottleneck_index)
    print(model)
    model.trainModel(data, args.epochs)
    model.train(False)
    with torch.no_grad():
        encoded = model.encode(data).cpu()
    report(f"Final:", encoded, data.cpu(), level=2)
    save_keys_pickle(encoded, args.keys_out)
