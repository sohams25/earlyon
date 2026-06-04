"""Small shared helpers: dataset loaders, checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models import (
    cifar_resnet_ee,
    efficientnet_b0_ee,
    mobilenetv2_ee,
    resnet18_ee,
    resnet50_ee,
)

_Batch = tuple[torch.Tensor, torch.Tensor]
_CIFAR_PREFIX = "cifar_resnet"

FACTORIES = {
    "resnet18": resnet18_ee,
    "resnet50": resnet50_ee,
    "mobilenetv2": mobilenetv2_ee,
    "efficientnet_b0": efficientnet_b0_ee,
}


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    # cifar_resnet_ee records its depth in the backbone string (e.g.
    # "cifar_resnet20"); reconstruct the architecture from that so save/load
    # round-trips. Without this, any cifar_resnet checkpoint is unloadable.
    if backbone.startswith(_CIFAR_PREFIX):
        suffix = backbone[len(_CIFAR_PREFIX) :]
        if not suffix.isdigit():
            raise ValueError(
                f"malformed cifar_resnet backbone {backbone!r}; expected e.g. 'cifar_resnet20'"
            )
        return cifar_resnet_ee(num_classes=num_classes, depth=int(suffix))
    if backbone not in FACTORIES:
        raise ValueError(
            f"unknown backbone {backbone!r}; choose from {list(FACTORIES)} or 'cifar_resnet<depth>'"
        )
    return FACTORIES[backbone](num_classes=num_classes, pretrained=pretrained)


def save_wrapper(model: EarlyExitWrapper, path: str | Path) -> None:
    """Save state_dict + config. Config is needed to rebuild the wrapper.

    ``routing_policy`` and ``entropy_thresholds`` are persisted too: without
    them an entropy-routed model would silently reload as confidence-routed
    (the wrapper default), discarding the calibrated entropy thresholds.
    """
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": model.config.backbone,
            "num_classes": model.config.num_classes,
            "routing_policy": model.config.routing_policy,
            "confidence_thresholds": list(model.config.confidence_thresholds),
            "entropy_thresholds": list(model.config.entropy_thresholds),
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

    # Validate config shape BEFORE mutating model.config. A checkpoint saved
    # against a different exit count would otherwise leave model.config in a
    # half-updated state if load_state_dict raised later.
    n_exits = len(model.config.exit_points)
    thr_len = len(cfg["confidence_thresholds"])
    weight_len = len(cfg["loss_weights"])
    if thr_len != n_exits:
        raise ValueError(
            f"checkpoint {Path(path).name}: confidence_thresholds has "
            f"length {thr_len}, but backbone {cfg['backbone']!r} has "
            f"{n_exits} exit points"
        )
    if weight_len != n_exits + 1:
        raise ValueError(
            f"checkpoint {Path(path).name}: loss_weights has length "
            f"{weight_len}, expected {n_exits + 1} (one per exit plus final)"
        )
    # entropy_thresholds + routing_policy are optional for back-compat with
    # pre-0.2 checkpoints; validate them only when present. load_wrapper
    # bypasses EarlyExitConfig.__post_init__, so re-validate the persisted policy
    # here — otherwise a corrupted checkpoint would silently mis-route.
    if "entropy_thresholds" in cfg and len(cfg["entropy_thresholds"]) != n_exits:
        raise ValueError(
            f"checkpoint {Path(path).name}: entropy_thresholds has length "
            f"{len(cfg['entropy_thresholds'])}, but backbone {cfg['backbone']!r} "
            f"has {n_exits} exit points"
        )
    policy = cfg.get("routing_policy", model.config.routing_policy)
    if policy not in {"confidence", "entropy"}:
        raise ValueError(
            f"checkpoint {Path(path).name}: unsupported routing_policy {policy!r} "
            "(allowed: 'confidence', 'entropy')"
        )
    if policy == "entropy" and "entropy_thresholds" not in cfg:
        raise ValueError(
            f"checkpoint {Path(path).name}: routing_policy='entropy' but no "
            "entropy_thresholds were persisted — the model would silently route "
            "on uncalibrated defaults"
        )

    model.config.confidence_thresholds = list(cfg["confidence_thresholds"])
    model.config.loss_weights = list(cfg["loss_weights"])
    model.config.temperature = float(cfg["temperature"])
    # back-compat: older checkpoints predate these fields; keep the fresh
    # model's defaults when the key is absent.
    model.config.routing_policy = policy
    if "entropy_thresholds" in cfg:
        model.config.entropy_thresholds = list(cfg["entropy_thresholds"])
    model.load_state_dict(payload["state_dict"])
    return model


def cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 2,
    val_split: float = 0.1,
) -> Tuple[DataLoader[_Batch], DataLoader[_Batch], DataLoader[_Batch]]:
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
