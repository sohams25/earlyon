"""Shared helpers for the model factories."""

from __future__ import annotations

import torch


def identity(x: torch.Tensor) -> torch.Tensor:
    """Pass-through final classifier for backbones whose ``forward`` already
    returns logits (torchvision ResNet/MobileNetV2/EfficientNet, CifarResNet)."""
    return x
