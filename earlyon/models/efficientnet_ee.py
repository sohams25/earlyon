"""Early-exit wrapper for torchvision EfficientNet-B0.

The block structure is `features[0..8]`. Exits at features[3] and features[5]
give a reasonable spread of compute across the network.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper


def _identity(x):
    return x


# Channel counts at the exit points (from the efficientnet-b0 spec)
_EFFNET_B0_CHANNELS = {"features.3": 40, "features.5": 112}


def efficientnet_b0_ee(num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    backbone = efficientnet_b0(weights="DEFAULT" if pretrained else None)
    if num_classes != 1000:
        in_feat = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feat, num_classes)

    exit_points = [
        ExitPoint("e0", "features.3", _EFFNET_B0_CHANNELS["features.3"]),
        ExitPoint("e1", "features.5", _EFFNET_B0_CHANNELS["features.5"]),
    ]
    heads = {
        ep.name: EarlyExitHead(ep.in_channels, num_classes, hidden_dim=128)
        for ep in exit_points
    }
    cfg = EarlyExitConfig(
        backbone="efficientnet_b0",
        num_classes=num_classes,
        exit_points=exit_points,
        confidence_thresholds=[0.85, 0.80],
        loss_weights=[0.2, 0.3, 0.5],
    )
    return EarlyExitWrapper(backbone, heads, _identity, cfg)
