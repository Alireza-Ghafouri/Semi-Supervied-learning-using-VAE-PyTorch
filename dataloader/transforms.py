from torchvision.transforms import Compose, RandomAffine, Grayscale, ToTensor, Normalize, Resize, ColorJitter
from torchvision.transforms import RandomApply, RandomRotation, RandomResizedCrop, RandomPerspective
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
# SVHN
svhn_transform_base= Compose([
            Resize(64),
            ToTensor(),
        ])

svhn_transform_1 = Compose([
    Resize(64),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ToTensor(),
    # Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
])

svhn_transform_2 = Compose([
    Resize(64),
    RandomApply([ColorJitter(brightness=0.5, contrast=0.5)], p=0.3),
    Grayscale(num_output_channels=3),
    ToTensor(),
    # Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
])

svhn_transform_3 = Compose([
    Resize(64),
    RandomApply([ColorJitter(brightness=0.5, contrast=0.5)], p=0.3),
    RandomRotation(degrees=15),
    ToTensor(),
    # Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
])

svhn_transform_4 = Compose([
    Resize(64),
    ToTensor(),
    RandomRotation(degrees=15),
    # Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
])

svhn_transform_5 = Compose([
    RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2)),  # Scaling between 80% and 120%
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change brightness, contrast, and saturation
    RandomPerspective(distortion_scale=0.5),  # perspective transformations
    ToTensor(),
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

svhn_transform_auto= Compose([
            Resize(64), 
            AutoAugment(policy=AutoAugmentPolicy.SVHN),  # Apply AutoAugment
            ToTensor(),
        ])


# CIFAR10
cifar10_transform_base= Compose([
            Resize(64), 
            ToTensor(),
        ])