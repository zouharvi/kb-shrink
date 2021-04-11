#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import l2_sim, mrr, read_keys_pickle, save_keys_pickle, DEVICE, vec_sim_order
import argparse
import numpy as np
import torch
import torch.nn as nn

order_old_ip = None
order_old_l2 = None


def report(prefix, encoded, data, level, order_old_ip=None, order_old_l2=None):
    if level >= 2:
        if order_old_ip is None:
            order_old_ip = vec_sim_order(data, sim_func=np.inner)

    if level >= 3:
        if order_old_l2 is None:
            order_old_l2 = vec_sim_order(data, sim_func=l2_sim)

    # V^2 similarity computations is computationally expensive, skip if not necessary
    if level == 3:
        order_new = vec_sim_order(encoded, sim_func=np.inner)
        mrr_val_ip = mrr(order_old_ip, order_new, 20, report=False)
        order_new = vec_sim_order(encoded, sim_func=l2_sim)
        mrr_val_l2 = mrr(order_old_l2, order_new, 20, report=False)
        print(f'{prefix} mrr_ip: {mrr_val_ip:.3f}, mrr_l2: {mrr_val_l2:.3f}')
        return mrr_val_ip, mrr_val_l2
    elif level == 2:
        order_new = vec_sim_order(encoded)
        mrr_val_ip = mrr(order_old_ip, order_new, 20, report=False)
        print(f'{prefix} mrr_ip: {mrr_val_ip:.3f}')
        return mrr_val_ip
    elif level == 1:
        print(f'{prefix}')


class Autoencoder(nn.Module):
    def __init__(self, model, bottleneck_width, batchSize=128, learningRate=0.001):
        super().__init__()

        if model == 1:
            self.layers = [
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
        elif model == 2:
            self.layers = [
                nn.Linear(768, 512),               # 1
                nn.Identity(),                     # 2
                nn.Linear(512, 256),               # 3
                nn.Identity(),                     # 4
                nn.Linear(256, bottleneck_width),  # 5
                nn.Identity(),                     # 6
                nn.Linear(bottleneck_width, 256),  # 7
                nn.Identity(),                     # 8
                nn.Linear(256, 512),               # 9
                nn.Identity(),                     # 10
                nn.Linear(512, 768),               # 11
                nn.Identity(),                     # 12
            ]
        elif model == 3:
            self.layers = [
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
            ]
        elif model == 4:
            self.layers = [
                nn.Linear(768, 512),               # 1
                nn.Tanh(),                         # 2
                nn.Linear(512, 256),               # 3
                nn.Tanh(),                         # 4
                nn.Linear(256, bottleneck_width),  # 5
                nn.Tanh(),                         # 6
                nn.Linear(bottleneck_width, 768),  # 7
            ]
        else:
            raise Exception("Unknown model specified")
        self.all_layers = nn.Sequential(*self.layers)

        self.model = model
        self.batchSize = batchSize
        self.learningRate = learningRate

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )
        self.criterion = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x):
        return self.all_layers(x)

    def encode(self, x, bottleneck_index):
        encoder = nn.Sequential(*self.layers[:bottleneck_index])
        return encoder(x)

    def decode(self, x, bottleneck_index):
        decoder = nn.Sequential(*self.layers[bottleneck_index:])
        return decoder(x)

    def trainModel(self, data, epochs, bottleneck_index, loglevel):
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

                if self.model == 4:
                    l1_penalty = 0.0001 * sum([p.abs().sum() for p in self.all_layers[-1].parameters()])
                    loss += l1_penalty
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = self.encode(data, bottleneck_index).cpu()

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.7f},",
                    encoded, data.cpu(), level=loglevel
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
        help='Which model to use')
    parser.add_argument(
        '--bottleneck-width', default=256, type=int,
        help='Dimension of the bottleneck layer')
    parser.add_argument(
        '--bottleneck-index', default=5, type=int,
        help='Position of the last encoder layer')
    parser.add_argument(
        '--epochs', default=1000, type=int)
    parser.add_argument(
        '--loglevel', default=1, type=int,
        help='Level at which to report')
    parser.add_argument(
        '--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = read_keys_pickle(args.keys_in)
    data = torch.Tensor(data).to(DEVICE)
    model = Autoencoder(args.model, args.bottleneck_width)
    print(model)
    model.trainModel(data, args.epochs, bottleneck_index=-1, loglevel=args.loglevel)
    model.train(False)
    with torch.no_grad():
        encoded = model.encode(data, args.bottleneck_index).cpu()
    report(f"Final:", encoded, data.cpu(), level=3)
    save_keys_pickle(encoded, args.keys_out)
