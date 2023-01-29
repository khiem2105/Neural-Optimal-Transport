import torch
from torch.utils.data import Dataset

from src.distributions import Sampler

class ToyDataset(Dataset):
    def __init__(self, sampler: Sampler, n_samples: int):
        super(ToyDataset, self).__init__()

        self.data = sampler.sample(n_samples)
        self.len = n_samples
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], 0