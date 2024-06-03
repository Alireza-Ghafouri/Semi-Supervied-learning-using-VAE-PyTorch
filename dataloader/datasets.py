from torchvision.datasets import SVHN, CIFAR10
from torch.utils.data import Dataset

class SVHNDataset(Dataset):
    def __init__(self, mode, transform= None):
        super(SVHNDataset, self).__init__()
        self.data = SVHN(root='./data', split=mode, download=True)
        self.transform = transform

    def __getitem__(self, index):

        image, label = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index

    def __len__(self):
        return len(self.data)

class CIFAR10Dataset(Dataset):
    def __init__(self, is_train, transform= None):
        super(CIFAR10Dataset, self).__init__()
        self.data = trainset = CIFAR10(root='./data', train= is_train, download=True)
        self.transform = transform

    def __getitem__(self, index):

        image, label = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index

    def __len__(self):
        return len(self.data)
    
class MyDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], idx