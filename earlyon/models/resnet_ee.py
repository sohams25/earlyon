"""Early-exit wrappers around torchvision ResNet18/ResNet50.

Exit placement follows the BranchyNet/ACM 2024 survey conventions:
- ResNet50: after layer1, layer2, layer3 (3 exits before final)
- ResNet18: after layer2, layer3 (2 exits before final)

The torchvision ResNet's ``forward`` returns logits, so ``final_classifier``
is identity.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet50

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper

# torchvision ResNet block output channels (BasicBlock for r18, Bottleneck for r50)
_R18_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
_R50_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}


def _identity(x):
    return x


def resnet50_ee(num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    backbone = resnet50(weights="DEFAULT" if pretrained else None)
    if num_classes != 1000:
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    exit_points = [
        ExitPoint("e0", "layer1", _R50_CHANNELS["layer1"]),
        ExitPoint("e1", "layer2", _R50_CHANNELS["layer2"]),
        ExitPoint("e2", "layer3", _R50_CHANNELS["layer3"]),
    ]
    heads = {
        ep.name: EarlyExitHead(ep.in_channels, num_classes, hidden_dim=256)
        for ep in exit_points
    }
    cfg = EarlyExitConfig(
        backbone="resnet50",
        num_classes=num_classes,
        exit_points=exit_points,
        confidence_thresholds=[0.85, 0.80, 0.75],
        loss_weights=[0.1, 0.2, 0.3, 0.4],
    )
    return EarlyExitWrapper(backbone, heads, _identity, cfg)


def resnet18_ee(num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    backbone = resnet18(weights="DEFAULT" if pretrained else None)
    if num_classes != 1000:
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    exit_points = [
        ExitPoint("e0", "layer2", _R18_CHANNELS["layer2"]),
        ExitPoint("e1", "layer3", _R18_CHANNELS["layer3"]),
    ]
    heads = {
        ep.name: EarlyExitHead(ep.in_channels, num_classes)
        for ep in exit_points
    }
    cfg = EarlyExitConfig(
        backbone="resnet18",
        num_classes=num_classes,
        exit_points=exit_points,
        confidence_thresholds=[0.85, 0.80],
        loss_weights=[0.2, 0.3, 0.5],
    )
    return EarlyExitWrapper(backbone, heads, _identity, cfg)
