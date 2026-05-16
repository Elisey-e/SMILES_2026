"""
5_4 — deterministic class-balanced train loader.

Goal:
  Reduce mini-batch class-prior noise for zero-order optimization.
  Especially useful when optimizer is weight-only and bias is not tuned.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

from augmentation import get_transforms


def _make_balanced_order(targets, batch_size: int, budget: int = 8192, n_classes: int = 100):
    buckets = defaultdict(list)
    for idx, y in enumerate(targets):
        buckets[int(y)].append(idx)

    ptr = {c: 0 for c in range(n_classes)}
    order = []

    n_batches = math.ceil(budget / batch_size)

    for b in range(n_batches):
        # Shift class cycle every batch.
        # For batch_size=64, each batch has 64 distinct classes.
        # Across 128 batches, all classes are visited many times.
        start = (b * batch_size) % n_classes

        for j in range(batch_size):
            c = (start + j) % n_classes
            bucket = buckets[c]
            idx = bucket[ptr[c] % len(bucket)]
            ptr[c] += 1
            order.append(idx)

            if len(order) >= budget:
                return order

    return order[:budget]


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    base_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    indices = _make_balanced_order(
        targets=base_dataset.targets,
        batch_size=batch_size,
        budget=8192,
        n_classes=100,
    )

    train_dataset = Subset(base_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, train_loader