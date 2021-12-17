import torch
import sys; sys.path.append("src")
from misc.retrieval_utils import DEVICE

class TorchPCA:
	def __init__(self, n_components):
		self.matrix = None
		self.n_components = n_components

	def fit(self, X):
		X = torch.tensor(X).to(DEVICE)
		U,S,V = torch.svd(X)
		eigvecs=U.t()[:,:self.n_components] # the first k vectors will be kept
		y=torch.mm(U,eigvecs)

		# save variables to the class object, the eigenpair and the centered data
		self.eigenpair = (eigvecs, S)
		self.matrix = y

	def transform(self, sample):
		return torch.mm(torch.tensor(sample).to(DEVICE), self.matrix)