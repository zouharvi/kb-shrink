import sys
sys.path.append("src")
from misc.utils import DEVICE, rprec_ip, rprec_l2
import numpy as np
import torch.nn as nn
import torch


def report(prefix, encoded, data):
    val_ip = rprec_ip(
        encoded["queries"], encoded["docs"],
        data["relevancy"], fast=True, report=False)
    val_l2 = rprec_l2(
        encoded["queries"], encoded["docs"],
        data["relevancy"], fast=True, report=False)
    print(f'{prefix} rprec_ip: {val_ip:.3f}, rprec_l2: {val_l2:.3f}')
    return val_ip, val_l2


def create_generator(data, batchSize, dataOrganization):
    # TODO: this will result in 5000, 5000, 1369 which is incorrect
    if dataOrganization == "dd":
        dataLoader1 = torch.utils.data.DataLoader(
            dataset=data["docs"], batch_size=batchSize, shuffle=True
        )
        dataLoader2 = torch.utils.data.DataLoader(
            dataset=data["docs"], batch_size=batchSize, shuffle=True
        )
        for x, y in zip(dataLoader1, dataLoader2):
            yield x, y, None
    elif dataOrganization == "qd":
        dataLoader1 = torch.utils.data.DataLoader(
            dataset=list(enumerate(data["queries"])), batch_size=batchSize, shuffle=True
        )
        dataLoader2 = torch.utils.data.DataLoader(
            dataset=list(enumerate(data["docs"])), batch_size=batchSize, shuffle=True
        )
        for (iquery, query), (idoc, doc) in zip(dataLoader1, dataLoader2):
            # TODO: this needs to be switched between IP and L2
            relevant = torch.ones(query.shape[0])
            for i, (i_iquery, i_idoc) in enumerate(zip(iquery, idoc)):
                if i_idoc in data["relevancy"][i_iquery]:
                    relevant[i] = 0.0
            yield query, doc, relevant.to(DEVICE)
    else:
        raise Exception("Generator not defined")

# TODO:  try lower learning rate


class ProjectionModel(nn.Module):
    # similarityGold relevancy
    # learningRate=0.00001
    # batchSize=5000

    # similarityGold ip, l2
    # learningRate=0.0001
    # batchSize=2500
    def __init__(self, model, dimension, batchSize=2500, dataOrganization="dd", similarityModel="l2", similarityGold="l2", merge=True, learningRate=0.0001):
        super().__init__()

        if model == 1:
            self.projection1 = nn.Linear(768, dimension)
            self.projection2 = nn.Linear(768, dimension)
        elif model == 2:
            self.projection1 = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Tanh(),
                nn.Linear(1024, 768),
                nn.Tanh(),
                nn.Linear(768, dimension),
                # This is optional. The final results are the same though the convergence is faster with this.
                nn.Tanh(),
            )
            self.projection2 = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Tanh(),
                nn.Linear(1024, 768),
                nn.Tanh(),
                nn.Linear(768, dimension),
                # This is optional. The final results are the same though the convergence is faster with this.
                nn.Tanh(),
            )
        else:
            raise Exception("Unknown model specified")

        if merge:
            self.projection2 = self.projection1

        self.batchSize = batchSize
        self.dataOrganization = dataOrganization
        self.learningRate = learningRate
        self.similarityGold = similarityGold

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )

        if similarityGold == "relevancy":
            if dataOrganization != "qd":
                raise Exception("Incompatible criterion and data organization")
            self.similarityGold = lambda d1, d2, relevancy: relevancy
            # TODO
        elif similarityGold == "l2":
            self.similarityGold = lambda d1, d2, relevancy: nn.PairwiseDistance(
                p=2)(d1, d2)
        elif similarityGold == "ip":
            self.similarityGold = lambda d1, d2, relevancy: torch.mul(d1, d2).sum(dim=1)
        else:
            raise Exception("Unknown similarity gold")

        if similarityModel == "l2":
            self.similarityModel = lambda d1, d2: nn.PairwiseDistance(
                p=2)(d1, d2)
        elif similarityModel == "ip":
            self.similarityModel = lambda d1, d2: torch.mul(d1, d2).sum(dim=1)
        else:
            raise Exception("Unknown similarity model")

        self.criterion = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x1, x2):
        y1 = self.projection1(x1)
        y2 = self.projection2(x2)
        out = self.similarityModel(y1, y2)
        return out

    def encode1(self, x):
        return self.projection1(x)

    def encode2(self, x):
        return self.projection2(x)

    def trainModel(self, data, epochs):
        for epoch in range(epochs):
            self.train(True)
            self.dataGenerator = create_generator(
                data, self.batchSize, self.dataOrganization)
            for sample1, sample2, sampleRelevancy in self.dataGenerator:
                # Predictions
                output = self(sample1, sample2)
                sample_sim = self.similarityGold(
                    sample1, sample2, sampleRelevancy
                )
                # Calculate Loss
                loss = self.criterion(output, sample_sim)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = {
                        "queries": self.encode1(data["queries"]).cpu().numpy(),
                        "docs": self.encode2(data["docs"]).cpu().numpy(),
                        "relevancy": data["relevancy"],
                    }

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.9f},",
                    encoded, data
                )
