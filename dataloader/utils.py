import numpy as np
from torch.utils.data import Subset
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from training.utils import apply_transformations

def split_data(full_trainset, num_labeled, num_classes, fold):
    np.random.seed(42)  # For reproducibility

    # Get indices of samples for each class
    class_indices = [[] for _ in range(num_classes)]
    for idx in range(len(full_trainset)):
        _, label, _ = full_trainset[idx]
        class_indices[label].append(idx)

    # Shuffle the indices within each class
    for class_idx in range(num_classes):
        np.random.shuffle(class_indices[class_idx])

    # Calculate the number of samples per class for the labeled subset
    num_labeled_per_class = num_labeled // num_classes

    # Select labeled indices
    labeled_indices = []
    for class_idx in range(num_classes):
        start_idx = fold * num_labeled_per_class
        end_idx = start_idx + num_labeled_per_class
        labeled_indices.extend(class_indices[class_idx][start_idx:end_idx])

    # Ensure the labeled_indices are sorted for consistency
    labeled_indices = sorted(labeled_indices)
    
    # Create the labeled subset
    labeled_trainset = Subset(full_trainset, labeled_indices)
    
    # Create the unlabeled subset by excluding the labeled indices
    all_indices = set(range(len(full_trainset)))
    unlabeled_indices = list(all_indices - set(labeled_indices))
    
    # Ensure the unlabeled_indices are sorted for consistency
    unlabeled_indices = sorted(unlabeled_indices)
    
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