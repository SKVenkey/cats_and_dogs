"""Data loading and preprocessing utilities for Cats vs Dogs classification."""

import os
from pathlib import Path
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return image transforms for training or validation/test."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def load_dataset(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Cats vs Dogs dataset from data_dir.
    Expects data_dir/train/ with class subdirectories: Cat/, Dog/
    Returns train, val, test DataLoaders.
    """
    full_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=get_transforms(train=True),
    )

    n_total = len(full_dataset)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val/test transforms (no augmentation)
    val_ds.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def preprocess_single_image(image_path: str) -> torch.Tensor:
    """Preprocess a single image file for inference. Returns (1, 3, 224, 224) tensor."""
    from PIL import Image
    transform = get_transforms(train=False)
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


CLASS_NAMES = ["Cat", "Dog"]
