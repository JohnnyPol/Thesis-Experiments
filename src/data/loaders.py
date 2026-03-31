from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms


def _build_normalize_transform(dataset_config: dict[str, Any]) -> transforms.Normalize:
    """
    Build normalization transform from dataset config.
    """
    norm_cfg = dataset_config.get("normalization", {})
    mean = norm_cfg.get("mean", [0.4914, 0.4822, 0.4465])
    std = norm_cfg.get("std", [0.2023, 0.1994, 0.2010])
    return transforms.Normalize(mean=mean, std=std)


def _build_original_transform(dataset_config: dict[str, Any]) -> transforms.Compose:
    """
    Build the default transform used for validation/testing and optionally
    for the non-augmented training branch.
    """
    input_cfg = dataset_config.get("input", {})
    image_size = input_cfg.get("image_size", 256)
    normalize = _build_normalize_transform(dataset_config)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def _build_augmentation_transform(dataset_config: dict[str, Any]) -> transforms.Compose:
    """
    Build the augmented training transform.
    """
    input_cfg = dataset_config.get("input", {})
    image_size = input_cfg.get("image_size", 256)
    normalize = _build_normalize_transform(dataset_config)

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.1),
    ])


def data_loader(
    data_dir: str,
    batch_size: int,
    random_seed: int = 42,
    valid_size: float = 0.1,
    shuffle: bool = True,
    test: bool = False,
    num_workers: int = 0,
    dataset_config: dict[str, Any] | None = None,
):
    """
    Build CIFAR-10 train/validation or test dataloaders.

    Args:
        data_dir: Dataset root path.
        batch_size: Batch size.
        random_seed: Random seed for train/valid split.
        valid_size: Fraction of training data used for validation.
        shuffle: Whether to shuffle indices before splitting.
        test: If True, return test loader only.
        num_workers: DataLoader num_workers.
        dataset_config: Optional dataset YAML contents.

    Returns:
        If test=True:
            DataLoader
        else:
            (train_loader, valid_loader)
    """
    dataset_config = dataset_config or {}

    download = dataset_config.get("download", True)
    splits_cfg = dataset_config.get("splits", {})
    loader_cfg = dataset_config.get("loader", {})

    if valid_size is None:
        valid_size = splits_cfg.get("valid_size", 0.1)
    if random_seed is None:
        random_seed = splits_cfg.get("random_seed", 42)

    transform_original = _build_original_transform(dataset_config)
    transform_aug = _build_augmentation_transform(dataset_config)

    if test:
        test_shuffle = dataset_config.get("loader", {}).get("shuffle", False)
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=transform_original,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=test_shuffle if shuffle is None else shuffle,
            num_workers=num_workers,
        )

    train_dataset_orig = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_original,
    )

    train_dataset_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_aug,
    )

    num_train = len(train_dataset_orig)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_dataset = ConcatDataset([
        torch.utils.data.Subset(train_dataset_orig, train_idx),
        torch.utils.data.Subset(train_dataset_aug, train_idx),
    ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_dataset = torch.utils.data.Subset(train_dataset_orig, valid_idx)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader