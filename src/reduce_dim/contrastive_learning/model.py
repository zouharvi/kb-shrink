raise NotImplementedError("Not adapted to new data orgnization (docs and queries as tuples)")

import sys; sys.path.append("src")
from misc.load_utils import center_data, norm_data
from misc.retrieval_utils import DEVICE, order_l2, rprec_ip, rprec_l2
import numpy as np
import torch.nn as nn
import torch

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

def create_generator(data, batchSize, dataOrganization):
    if dataOrganization == "qd2":
        query_order_all = order_l2(data["queries"].cpu(), data["docs"].cpu(), [2] * len(data["queries"]), fast=True)
        dataLoader1 = torch.utils.data.DataLoader(
            dataset=list(zip(data["queries"], query_order_all)), batch_size=1, shuffle=True
        )

        for query, query_order in dataLoader1:
            random_docs = data["docs"].cpu()[np.random.choice(np.arange(len(data["docs"])), 64)]
            close_docs = data["docs"][query_order[0]]
            yield query.to(DEVICE), close_docs.to(DEVICE), random_docs.to(DEVICE)
    else:
        raise Exception("Generator not defined")

class ContrastiveLearningModel(nn.Module):
    def __init__(self, model, dimension, batchSize, learningRate, dataOrganization, merge, similarityModel):
        super().__init__()
        
        if model == 1:
            projection_builder = lambda: nn.Linear(768, dimension)
        elif model == 2:
            projection_builder = lambda: nn.Sequential(
                nn.Linear(768, 1024),
                nn.Tanh(),
                nn.Linear(1024, 768),
                nn.Tanh(),
                nn.Linear(768, dimension),
                # This is optional. The final results are the same though the convergence is faster with this.
                # nn.Tanh(),
            )
        else:
            raise Exception("Unknown model specified")

        if merge:
            self.projection1 = projection_builder()
            self.projection2 = self.projection1
        else:
            self.projection1 = projection_builder()
            self.projection2 = projection_builder()


        self.batchSize = batchSize
        self.dataOrganization = dataOrganization
        self.learningRate = learningRate

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )

        if similarityModel == "l2":
            self.similarityModel = \
                lambda query, docs: -nn.PairwiseDistance(p=2)(query.repeat((docs.shape[0], 1)), docs)
        elif similarityModel == "ip":
            self.similarityModel = \
                lambda query, docs: torch.mul(query.repeat((docs.shape[0], 1)), docs).sum(dim=1)
        else:
            raise Exception("Unknown similarity model")

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

    def trainModel(self, data, epochs, post_cn):
        for epoch in range(epochs):
            self.train(True)
            self.dataGenerator = create_generator(
                data, self.batchSize, self.dataOrganization)

            iteration = 0
            loss = 0.0
            for query, sample_pos, sample_neg in self.dataGenerator:
                # Predictions
                output_pos = self(query, sample_pos)
                output_neg = self(query, sample_neg)
                # Calculate Loss
                sum_pos = torch.sum(torch.exp(-1*output_pos))
                sum_neg = torch.sum(torch.exp(-1*output_neg))
                loss += sum_pos / sum_neg

                iteration += 1
                if iteration == self.batchSize:
                    iteration = 0
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            if (epoch + 1) % 1 == 0:
                self.train(False)
                with torch.no_grad():
                    encoded = {
                        "queries": self.encode1(data["queries"]).cpu().numpy(),
                        "docs": self.encode2(data["docs"]).cpu().numpy(),
                        "relevancy": data["relevancy"],
                    }

                report(
                    f"epoch [{epoch+1}/{epochs}], loss_l2: {loss.data:.9f},",
                    encoded, data, post_cn
                )
