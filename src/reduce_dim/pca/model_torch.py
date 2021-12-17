import torch
import sys; sys.path.append("src")
from misc.retrieval_utils import DEVICE

class TorchPCA:
	def __init__(self, n_components):
		self.matrix = None
		self.n_components = n_components

	def fit(self, X):
		X = torch.tensor(X)
		U,S,V = torch.svd(X)
		self.matrix = U.t()[:,:self.n_components].to(DEVICE)

		return self

	def transform(self, sample):
		return torch.mm(torch.tensor(sample).to(DEVICE), self.matrix)