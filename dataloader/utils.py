import numpy as np
from torch.utils.data import Subset
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from training.utils import apply_transformations

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

def create_pseudo_labeled_dataset(net, unlabeled_trainloader, mannual_dataset, device, transforms, confidence_threshold):
    pseudo_labeled_images = []
    pseudo_labels = []

    net.eval()
    with torch.no_grad():
        for data in tqdm(unlabeled_trainloader, desc="Pseudo Labelling"):
            
            copy_images, _, _ = data
            images, _, _ = apply_transformations(data, transforms_list= transforms)

            images= images.to(device)
            
            logits = net(images)[-1]
            # predicted_labels = torch.argmax(logits, 1)
            
            # pseudo_labeled_images.extend(copy_images)
            # pseudo_labels.extend(predicted_labels.tolist())

            probabilities = F.softmax(logits, dim=1)
            max_probs, predicted_labels = torch.max(probabilities, dim=1)
            
            for img, label, prob in zip(copy_images, predicted_labels, max_probs):
                if prob >= confidence_threshold:
                    pseudo_labeled_images.append(img)
                    pseudo_labels.append(int(label))

    # Convert pseudo-labels to int
    # pseudo_labels = [int(label) for label in pseudo_labels]

    # Create pseudo-labeled dataset
    pseudo_labeled_trainset = mannual_dataset(pseudo_labeled_images, pseudo_labels)
    print(f'\n %{round( 100 * len(pseudo_labeled_trainset) / len(unlabeled_trainloader.dataset) , 1)} of unlabeled samples pseudo labeled!\n')
    return pseudo_labeled_trainset

def selective_collate(batch):
    original_images, labels, indexes = zip(*batch)
    # labels = torch.tensor(labels)
    # indexes = torch.tensor(indexes)
    return  original_images, labels, indexes