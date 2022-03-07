#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import mean_squared_error as mse
import torch.nn as nn
import torch
from misc.load_utils import center_data, norm_data
from misc.retrieval_utils import DEVICE, rprec_a_ip, rprec_a_l2


def summary_performance(prefix, data_reduced, data, post_cn):
    if post_cn:
        data_reduced = center_data(data_reduced)
        data_reduced = norm_data(data_reduced)

    val_l2 = rprec_a_l2(
        data_reduced["queries"],
        data_reduced["docs"],
        data["relevancy"],
        data["relevancy_articles"],
        data["docs_articles"],
        fast=True
    )
    if post_cn:
        val_ip = val_l2
    else:
        val_ip = rprec_a_ip(
            data_reduced["queries"],
            data_reduced["docs"],
            data["relevancy"],
            data["relevancy_articles"],
            data["docs_articles"],
            fast=True
        )
    print(f'{prefix} rprec_ip: {val_ip:.3f}, rprec_l2: {val_l2:.3f}')
    return val_ip, val_l2


def get_train_data(data, train_key, train_crop_n):
    if train_key == "q":
        return data["queries"][:train_crop_n]
    elif train_key == "d":
        return data["docs"][:train_crop_n]
    elif train_key == "dq":
        return np.concatenate((data["docs"], data["queries"]))[:train_crop_n]
    else:
        raise Exception("Unknown train key")


class AutoencoderModel(nn.Module):
    # prev learningRate 0.001
    def __init__(self, model, bottleneck_width, batchSize=128, learningRate=0.001, seed=0):
        super().__init__()

        torch.manual_seed(seed)

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
                nn.Linear(bottleneck_width, bottleneck_width // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_width // 2, 768),
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
                nn.Conv1d(1, 768 // bottleneck_width,
                          2, stride=1, padding='same'),
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

    def encode_safe_native(self, data):
        dataLoader = torch.utils.data.DataLoader(
            dataset=data, batch_size=1024 * 128, shuffle=False
        )

        out = []
        for sample in dataLoader:
            out.append(self.encoder(sample.to(DEVICE)))

        return out

    def encode_safe(self, data):
        dataLoader = torch.utils.data.DataLoader(
            dataset=data, batch_size=1024 * 128, shuffle=False
        )

        out = []
        for sample in dataLoader:
            out += [x.cpu().numpy() for x in self.encoder(sample.to(DEVICE))]

        return out

    def decode_safe(self, data):
        dataLoader = torch.utils.data.DataLoader(
            dataset=data, batch_size=1024 * 128, shuffle=False
        )

        out = []
        for sample in dataLoader:
            out += [x.cpu().numpy() for x in self.decoder(sample.to(DEVICE))]

        return out

    def decode(self, x):
        return self.decoder(x)

    def train_routine(self, data, data_train, epochs, post_cn, regularize, train_key="docs", skip_eval=False, train_crop_n=None):
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=get_train_data(data_train, train_key, train_crop_n), batch_size=self.batchSize, shuffle=True
        )

        for epoch in range(epochs):
            self.train(True)
            for sample in self.dataLoader:
                # predictions
                sample = sample.to(DEVICE)
                output = self(sample)
                # calculate Loss
                loss = self.criterion(output, sample)

                # backpropagation
                self.optimizer.zero_grad()

                if regularize:
                    # L1
                    regularization_loss = 10**(-8) * sum(
                        [p.abs().sum() for p in self.decoder.parameters()]
                    )
                    loss += regularization_loss
                loss.backward()
                self.optimizer.step()

            if not skip_eval:
                # in some cases do not eval after every epoch
                self.eval_routine(
                    data,
                    prefix=f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.7f},",
                    post_cn=post_cn)

    def eval_routine(self, data, data_orig, scaler_center, scaler_norm, post_cn, prefix=""):
        self.train(False)
        with torch.no_grad():
            queries_data_enc = self.encode_safe(data["queries"])
            docs_data_enc = self.encode_safe(data["docs"])
            queries_data_dec = self.decode_safe(queries_data_enc)
            docs_data_dec = self.decode_safe(docs_data_enc)
            encoded = {
                "queries": queries_data_enc,
                "docs": docs_data_enc,
            }
            decoded = {
                "queries": queries_data_dec,
                "docs": docs_data_dec,
            }
            decoded = scaler_center.inverse_transform(scaler_norm.inverse_transform(decoded))

            loss_q = mse(
                data_orig["queries"],
                decoded["queries"]
            )
            # loss of only the first 10k documents because it has to get copied
            loss_d = mse(
                data_orig["docs"][:10000],
                decoded["docs"][:10000]
            )


        val_ip, val_l2 = summary_performance(
            prefix=prefix,
            data_reduced=encoded,
            data=data,
            post_cn=post_cn
        )
        return val_ip, val_l2, loss_q, loss_d

    def eval_routine_no_loss(self, data, post_cn, prefix=""):
        self.train(False)
        with torch.no_grad():
            queries_data = self.encode_safe(data["queries"])
            docs_data = self.encode_safe(data["docs"])
            encoded = {
                "queries": queries_data,
                "docs": docs_data,
            }

        val_ip, val_l2 = summary_performance(
            prefix=prefix,
            data_reduced=encoded,
            data=data,
            post_cn=post_cn
        )
        return val_ip, val_l2
