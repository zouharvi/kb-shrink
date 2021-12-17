import torch
from misc.retrieval_utils import DEVICE

class TorchPCA:
	def __init__(self):
		self.matrix = None

	def fit(self, X, k):
		X = torch.tensor(X).to(DEVICE)
		U,S,V = torch.svd(X)
		eigvecs=U.t()[:,:k] # the first k vectors will be kept
		y=torch.mm(U,eigvecs)

		# save variables to the class object, the eigenpair and the centered data
		self.eigenpair = (eigvecs, S)
		self.matrix = y

	def transform(self, sample):
		return torch.mm(torch.tensor(sample).to(DEVICE), self.matrix)