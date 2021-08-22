#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import center_data, norm_data
from misc.retrieval_utils import DEVICE, rprec_ip, rprec_l2
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

class SimDistillationFromAutoencoderModel(nn.Module):
    def __init__(self, model, dimension, batchSize=128, learningRate=0.001):
        super().__init__()

        if model == 1:
            self.encoder = nn.Linear(768, dimension)
            self.decoder = nn.Linear(dimension, 768)
        else:
            raise Exception("Unknown model specified")

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
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def trainModel(self, data, epochs, post_cn):
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=torch.cat((data["docs"],data["queries"])), batch_size=self.batchSize, shuffle=True
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

            if (epoch + 1) % 1 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = {
                        "queries": self.encode(data["queries"]).cpu().numpy(),
                        "docs": self.encode(data["docs"]).cpu().numpy(),
                    }

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.7f},",
                    encoded, data, post_cn
                )
