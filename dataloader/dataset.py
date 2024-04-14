import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import Dataset



class SVHNDataset(Dataset):
    def __init__(self, mode):
        super(SVHNDataset, self).__init__()
        self.data = SVHN(
                        root='./data',
                        split=mode,
                        download=True)
        self.calculate_mean_std()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __getitem__(self, index):
    
        image, label = self.data[index]
        
        # Applying transforms on image
        image = self.transform(image)
 
            
        return image, label

    def __len__(self):
        return len(self.data)
    
    def calculate_mean_std(self):
        mean = torch.zeros(3)
        std = torch.zeros(3)
        num_samples = len(self.data)

        for i in range(num_samples):
            image, _ = self.data[i]
            mean += torch.mean(transforms.ToTensor()(image), dim=(1, 2))
            std += torch.std(transforms.ToTensor()(image), dim=(1, 2))

        mean /= num_samples
        std /= num_samples

        self.mean = mean
        self.std = std 