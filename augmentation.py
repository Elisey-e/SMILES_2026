import torchvision.transforms as T


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose(
            [
                # Mild CIFAR-style spatial noise before upscaling.
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(p=0.5),

                # Match validation resolution.
                T.Resize(224, antialias=True),

                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )

    return T.Compose(
        [
            T.Resize(224, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )