import torch
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, 
        #    mean, 
        #    std,
           ):

    # Unnormalize the images
    # img = img * std[:, None, None] + mean[:, None, None]

    # Convert torch tensor to numpy array
    npimg = img.numpy()

    # Transpose the image array to match matplotlib format
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Display the image
    # plt.show()

def apply_transformations(data, transforms_list):
    aug_labels= []
    aug_indexes= []
    aug_images = []
    images, labels, indexes = data
    for i, image in enumerate(images):
        for transform in transforms_list:
            aug_images.append(transform(image))
            aug_labels.append(labels[i])
            aug_indexes.append(indexes[i])

    images = torch.stack(aug_images)
    labels = torch.tensor(aug_labels)
    indexes = torch.tensor(aug_indexes)
    return images, labels, indexes

def linear_threshold_scheduler(start, end, num_phases, phase):
    return start + (end - start) * (phase / num_phases)

def exponential_threshold_scheduler(start, end, num_phases, phase):
    return start * ((end / start) ** (phase / num_phases))