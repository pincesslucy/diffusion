import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cv2

class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = os.listdir(path)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.dataset[idx]))
        img = img.convert('RGB')
        img = self.transform(img)
        return img
    
