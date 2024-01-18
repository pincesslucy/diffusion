import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        pass