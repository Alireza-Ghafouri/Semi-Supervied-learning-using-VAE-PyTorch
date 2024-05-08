import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import SVHN, CIFAR10
from torch.utils.data import Dataset

class SVHNDataset(Dataset):
    def __init__(self, mode):
        super(SVHNDataset, self).__init__()
        self.data = SVHN(root='./data', split=mode, download=True)
        self.transform = transforms.Compose([
            transforms.Resize(64),  # Resize images to 64x64
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        image, label = self.data[index]
        # Applying transforms on image
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

class CIFAR10Dataset(Dataset):
    def __init__(self, is_train):
        super(CIFAR10Dataset, self).__init__()
        self.data = trainset = CIFAR10(root='./data', train= is_train, download=True)
        self.transform = transforms.Compose([
            transforms.Resize(64),  # Resize images to 64x64
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        image, label = self.data[index]
        # Applying transforms on image
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)