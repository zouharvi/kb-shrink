#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_keys_pickle, save_keys_pickle, DEVICE, rprec_ip, rprec_l2, center_data, norm_data
import argparse
import numpy as np
import torch
import torch.nn as nn

def report(prefix, encoded, data, post_cn):
    if post_cn:
        encoded = center_data(encoded)
        encoded = norm_data(encoded)
    val_l2 = rprec_l2(
        encoded["queries"], encoded["docs"],
        data["relevancy"], fast=True, report=False)
    if post_cn:
        val_ip = val_l2
    else:
        val_ip = rprec_ip(
            encoded["queries"], encoded["docs"],
            data["relevancy"], fast=True, report=False)
    print(f'{prefix} rprec_ip: {val_ip:.3f}, rprec_l2: {val_l2:.3f}')
    return val_ip, val_l2

class AutoencoderModel(nn.Module):
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
        elif model == 5:
            self.layers = [
                nn.Linear(768, 1024),              # 1
                nn.Tanh(),                         # 2
                nn.Linear(1024, 1024),             # 3
                nn.Tanh(),                         # 4
                nn.Linear(1024, bottleneck_width), # 5
                nn.Tanh(),                         # 6
                nn.Linear(bottleneck_width, 768),  # 7
            ]
        elif model == 6:
            self.layers = [
                nn.Linear(768, bottleneck_width),  # 1
                nn.Linear(bottleneck_width, 768),  # 2
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

    def trainModel(self, data, epochs, bottleneck_index, post_cn):
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=data["docs"], batch_size=self.batchSize, shuffle=True
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
                    pass
                    # L1
                    # regularization_loss = 0.00001 * sum([p.abs().sum() for p in self.all_layers[-1].parameters()])
                    # L2
                    # regularization_loss = 0.00001 * sum([p.pow(2).sum() for p in self.all_layers[-1].parameters()])
                    # loss += regularization_loss
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = {
                        "queries": self.encode(data["queries"], bottleneck_index).cpu().numpy(),
                        "docs": self.encode(data["docs"], bottleneck_index).cpu().numpy(),
                    }

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.7f},",
                    encoded, data, post_cn
                )
