import sys
sys.path.append("src")
from misc.utils import DEVICE, rprec_ip, rprec_l2
import numpy as np
import torch.nn as nn
import torch

def report(prefix, encoded, data, level):
    if level == 2:
        acc_val_ip = rprec_ip(encoded["queries"], encoded["docs"], data["relevancy"], fast=True, report=False)
        acc_val_l2 = rprec_l2(encoded["queries"], encoded["docs"], data["relevancy"], fast=True, report=False)
        print(f'{prefix} acc_ip: {acc_val_ip:.3f}, acc_l2: {acc_val_l2:.3f}')
        return acc_val_ip, acc_val_l2
    elif level == 1:
        print(f'{prefix}')

def create_generator(data, batchSize, dataOrganization):
    # TODO: this will result in 5000, 5000, 1369 which is incorrect
    if dataOrganization == "dd":
        dataLoader1 = torch.utils.data.DataLoader(
            dataset=data["docs"], batch_size=batchSize, shuffle=True
        )
        dataLoader2 = torch.utils.data.DataLoader(
            dataset=data["docs"], batch_size=batchSize, shuffle=True
        )
        for x,y in zip(dataLoader1, dataLoader2):
            yield x,y,None
    elif dataOrganization == "qd":
        dataLoader1 = torch.utils.data.DataLoader(
            dataset=list(enumerate(data["queries"])), batch_size=batchSize, shuffle=True
        )
        dataLoader2 = torch.utils.data.DataLoader(
            dataset=list(enumerate(data["docs"])), batch_size=batchSize, shuffle=True
        )
        for (iquery,query),(idoc,doc) in zip(dataLoader1, dataLoader2):
            relevant = torch.zeros(query.shape[0])
            for i, (i_iquery, i_idoc) in enumerate(zip(iquery, idoc)):
                if i_idoc in data["relevancy"][i_iquery]:
                    relevant[i] = 1.0
            yield query,doc,relevant.to(DEVICE)
    else:
        raise Exception("Generator not defined")

# TODO:  try lower learning rate
class ProjectionModel(nn.Module):
    def __init__(self, model, dimension=64, batchSize=5000, dataOrganization="qd", similarityGold="relevancy", merge=True, learningRate=0.0001):
        super().__init__()

        if model == 1:
            self.projection1 = nn.Linear(768, dimension)
            self.projection2 = nn.Linear(768, dimension)
        elif model == 2:
            self.projection1 = nn.Sequential(
                nn.Linear(768, 512),
                nn.Tanh(),
                nn.Linear(512, dimension),
            )
            self.projection2 = nn.Sequential(
                nn.Linear(768, 512),
                nn.Tanh(),
                nn.Linear(512, dimension),
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
        
        self.similarity = nn.PairwiseDistance(p=2)

        if similarityGold == "relevancy":
            if dataOrganization != "qd":
                raise Exception("Incompatible criterion and data organization")
            self.similarityGold = lambda d1, d2, relevancy: relevancy
            # TODO
        elif similarityGold == "l2":
            self.similarityGold = lambda d1, d2, relevancy: self.similarity(d1, d2)

        self.criterion = nn.MSELoss()
        self.to(DEVICE)


    def forward(self, x1, x2):
        y1 = self.projection1(x1)
        y2 = self.projection1(x2)
        out = self.similarity(y1, y2)
        return out

    def encode1(self, x):
        return self.projection1(x)
    def encode2(self, x):
        return self.projection2(x)

    def trainModel(self, data, epochs, loglevel):
        for epoch in range(epochs):
            self.train(True)
            self.dataGenerator = create_generator(data, self.batchSize, self.dataOrganization)
            for sample1, sample2, sampleRelevancy in self.dataGenerator:
                # Predictions
                output = self(sample1, sample2)
                sample_sim = self.similarityGold(sample1, sample2, sampleRelevancy)
                # Calculate Loss
                loss = self.criterion(output, sample_sim)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 1000 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = {
                        "queries": self.encode1(data["queries"]).cpu().numpy(),
                        "docs": self.encode2(data["docs"]).cpu().numpy(), 
                        "relevancy": data["relevancy"], 
                    } 

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.9f},",
                    encoded, data, level=loglevel
                )
