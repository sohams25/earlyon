"""Early-exit wrapper around torchvision MobileNetV2.

Exit points sit inside the ``features`` sequential at the 3rd and 10th
inverted-residual blocks, accessed via dotted paths ``features.3`` and
``features.10``. Channel widths come from the standard MobileNetV2 spec.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def mobilenetv2_ee(num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    backbone = mobilenet_v2(weights="DEFAULT" if pretrained else None)
    if num_classes != 1000:
        in_feat = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feat, num_classes)

    # MobileNetV2 block output channels at the two exit points
    exit_points = [
        ExitPoint("e0", "features.3", 24),
        ExitPoint("e1", "features.10", 64),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, num_classes) for ep in exit_points}
    cfg = EarlyExitConfig(
        backbone="mobilenetv2",
        num_classes=num_classes,
        exit_points=exit_points,
        confidence_thresholds=[0.85, 0.80],
        loss_weights=[0.2, 0.3, 0.5],
    )
    return EarlyExitWrapper(backbone, heads, _identity, cfg)
