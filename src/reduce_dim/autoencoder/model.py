#!/usr/bin/env python3

raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.retrieval_utils import DEVICE, rprec_ip, rprec_l2
from misc.load_utils import center_data, norm_data
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
    # prev learningRate 0.001
    def __init__(self, model, bottleneck_width, batchSize=128, learningRate=0.001, skip_move=False):
        super().__init__()

        if model == 0:
            # learned PCA
            self.layers_enc = [
                nn.Linear(768, bottleneck_width),
                nn.Tanh(),                       
            ]
            self.layers_dec = [
                nn.Linear(bottleneck_width, 768),
                nn.Tanh(),                       
            ]
        elif model == 1:
            # learned PCA
            self.layers_enc = [
                nn.Linear(768, bottleneck_width),
            ]
            self.layers_dec = [
                nn.Linear(bottleneck_width, 768),
            ]
        elif model == 2:
            # symmetric
            self.layers_enc = [
                nn.Linear(768, 512),             
                nn.Tanh(),                       
                nn.Linear(512, 256),             
                nn.Tanh(),                       
                nn.Linear(256, bottleneck_width),
            ]
            self.layers_dec = [
                nn.Linear(bottleneck_width, 256),
                nn.Tanh(),                       
                nn.Linear(256, 512),             
                nn.Tanh(),                       
                nn.Linear(512, 768),             
            ]
        elif model == 3:
            # shallow decoder
            self.layers_enc = [
                nn.Linear(768, 512),             
                nn.Tanh(),                       
                nn.Linear(512, 256),             
                nn.Tanh(),                       
                nn.Linear(256, bottleneck_width),
            ]
            self.layers_dec = [
                nn.Linear(bottleneck_width, 768),
            ]
        elif model == 4:
            # double bottleneck
            self.layers_enc = [
                nn.Linear(768, 512),             
                nn.Tanh(),                       
                nn.Linear(512, 256),             
                nn.Tanh(),                       
                nn.Linear(256, bottleneck_width),
            ]
            self.layers_dec = [
                nn.Linear(bottleneck_width, bottleneck_width//2),
                nn.Tanh(),                       
                nn.Linear(bottleneck_width//2, 768),
            ]
        elif model == 5:
            # convolution upscaler
            self.layers_enc = [
                nn.Linear(768, 512),             
                nn.Tanh(),                       
                nn.Linear(512, 256),             
                nn.Tanh(),                       
                nn.Linear(256, bottleneck_width),
            ]
            self.layers_dec = [
                nn.Conv1d(1, 768//bottleneck_width, 2, stride=1, padding='same'),
                nn.Flatten()
            ]
        else:
            raise Exception("Unknown model specified")
        self.encoder = nn.Sequential(*self.layers_enc)
        self.decoder = nn.Sequential(*self.layers_dec)

        self.model = model
        self.batchSize = batchSize
        self.learningRate = learningRate

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )
        self.criterion = nn.MSELoss()
        if not skip_move:
            self.to(DEVICE)

    def forward(self, x):
        if self.model == 5:
            x = self.encoder(x)
            x = torch.reshape(x, (x.shape[0], 1, 64))
            return self.decoder(x)
        else:
            return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def trainModel(self, data, epochs, post_cn, regularize, skip_eval=False, train_crop_n=None):
        if train_crop_n is not None:
            self.dataLoader = torch.utils.data.DataLoader(
                dataset=data["docs"][:train_crop_n], batch_size=self.batchSize, shuffle=True
            )
        else:
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

                if regularize:
                    # L1
                    regularization_loss = 0.00000001 * sum([p.abs().sum() for p in self.decoder.parameters()])
                    loss += regularization_loss
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 5 == 0:
                if not skip_eval:
                    print("evaling")
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
                else:
                    print(
                        f"epoch [{epoch+1}/{epochs}]",
                    )