from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms


class CIFAR101(Dataset):
    """
    CIFAR-10.1 v6 test-only dataset stored as NumPy arrays.
    """

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    def __init__(
        self,
        root: str,
        train: bool = False,
        download: bool = False,
        transform: Any | None = None,
    ) -> None:
        if train:
            raise ValueError("CIFAR-10.1 is a test-only dataset.")

        root_path = Path(root)
        data_path = root_path / "cifar10.1_v6_data.npy"
        labels_path = root_path / "cifar10.1_v6_labels.npy"

        if not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "CIFAR-10.1 v6 files were not found. Expected "
                f"'{data_path}' and '{labels_path}'. Download them manually "
                "from the CIFAR-10.1 repository before running evaluation."
            )

        self.data = np.load(data_path)
        labels = np.load(labels_path).astype(int)

        if len(self.data) != len(labels):
            raise ValueError(
                "CIFAR-10.1 data/label length mismatch: "
                f"{len(self.data)} images and {len(labels)} labels."
            )

        self.targets = labels.tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index])
        label = int(self.targets[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


DATASET_BUILDERS = {
    "cifar10": datasets.CIFAR10,
    "cifar10_1": CIFAR101,
    "cifar100": datasets.CIFAR100,
}


def _resolve_dataset_builder(dataset_config: dict[str, Any]):
    dataset_name = str(dataset_config.get("name", "cifar10")).lower()
    if dataset_name not in DATASET_BUILDERS:
        supported = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of: {supported}."
        )
    return DATASET_BUILDERS[dataset_name]


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
    Build CIFAR train/validation or test dataloaders.

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
    dataset_builder = _resolve_dataset_builder(dataset_config)

    if test:
        test_shuffle = dataset_config.get("loader", {}).get("shuffle", False)
        dataset = dataset_builder(
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

    train_dataset_orig = dataset_builder(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_original,
    )

    train_dataset_aug = dataset_builder(
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
