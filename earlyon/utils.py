"""Small shared helpers: dataset loaders, checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models import efficientnet_b0_ee, mobilenetv2_ee, resnet18_ee, resnet50_ee

FACTORIES = {
    "resnet18": resnet18_ee,
    "resnet50": resnet50_ee,
    "mobilenetv2": mobilenetv2_ee,
    "efficientnet_b0": efficientnet_b0_ee,
}


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    if backbone not in FACTORIES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(FACTORIES)}")
    return FACTORIES[backbone](num_classes=num_classes, pretrained=pretrained)


def save_wrapper(model: EarlyExitWrapper, path: str | Path) -> None:
    """Save state_dict + config. Config is needed to rebuild the wrapper."""
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": model.config.backbone,
            "num_classes": model.config.num_classes,
            "confidence_thresholds": list(model.config.confidence_thresholds),
            "loss_weights": list(model.config.loss_weights),
            "temperature": model.config.temperature,
        },
    }
    torch.save(payload, Path(path))


def load_wrapper(path: str | Path, pretrained_backbone: bool = False) -> EarlyExitWrapper:
    # weights_only=True prevents pickle code-exec from malicious .pth files
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    cfg = payload["config"]
    model = build_model(cfg["backbone"], cfg["num_classes"], pretrained=pretrained_backbone)
    model.config.confidence_thresholds = list(cfg["confidence_thresholds"])
    model.config.loss_weights = list(cfg["loss_weights"])
    model.config.temperature = float(cfg["temperature"])
    model.load_state_dict(payload["state_dict"])
    return model


def cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 2,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for CIFAR-10.

    Images are upsampled to ``image_size`` so torchvision ImageNet-pretrained
    backbones work without modification. For honest benchmarking on CIFAR
    natively you'd want a CIFAR-specific ResNet variant; v0.1 prioritizes
    the pip-install story.
    """
    import torchvision
    from torchvision import transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf
    )
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)
    n_val = int(len(train_full) * val_split)
    indices = list(range(len(train_full)))
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    val_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=eval_tf
    )
    train_loader = DataLoader(
        Subset(train_full, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
