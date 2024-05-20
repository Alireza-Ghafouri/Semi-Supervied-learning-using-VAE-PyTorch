import numpy as np
from torch.utils.data import Subset
import torch
from tqdm.auto import tqdm

def split_data(full_trainset, labeled_ratio):

    np.random.seed(42)

    # Split the dataset into labeled and unlabeled subsets based on the defined ratio
    num_labeled = int(len(full_trainset) * labeled_ratio)
    indices = np.arange(len(full_trainset))
    np.random.shuffle(indices)

    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]

    # Create labeled and unlabeled datasets
    labeled_trainset = Subset(full_trainset, labeled_indices)
    unlabeled_trainset = Subset(full_trainset, unlabeled_indices)

    return labeled_trainset, unlabeled_trainset

def create_pseudo_labeled_dataset(net, unlabeled_trainloader, mannual_dataset, device):
    pseudo_labeled_images = []
    pseudo_labels = []

    net.eval()
    with torch.no_grad():
        for images, _ in tqdm(unlabeled_trainloader, desc="Pseudo Labelling"):
            images = images.to(device)
            
            logits = net(images)[-1]
            predicted_labels = torch.argmax(logits, 1)
            
            pseudo_labeled_images.extend(images)
            pseudo_labels.extend(predicted_labels.tolist())

    # Convert pseudo-labels to int
    pseudo_labels = [int(label) for label in pseudo_labels]
    pseudo_labeled_images = [sample.to('cpu') for sample in pseudo_labeled_images]


    # Create pseudo-labeled dataset
    pseudo_labeled_trainset = mannual_dataset(pseudo_labeled_images, pseudo_labels)

    return pseudo_labeled_trainset