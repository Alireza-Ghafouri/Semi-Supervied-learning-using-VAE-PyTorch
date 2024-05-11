import numpy as np
from torch.utils.data import Subset

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
