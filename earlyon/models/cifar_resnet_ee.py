"""CIFAR-native ResNet with three early exits.

The torchvision ResNets target 224x224 ImageNet input — applying them to
32x32 CIFAR inputs either upsamples (wasting compute) or destroys spatial
resolution in the first maxpool. He et al. 2015 (Sec. 4.2) describe a
distinct ResNet family for CIFAR:

* 3x3 stride-1 first conv (not 7x7 stride-2)
* no initial maxpool
* three stages of basic blocks at channel widths 16, 32, 64
* total depth = 6n + 2, where n is the per-stage block count

Exit points are attached at the end of each stage. This is a fair
counterpart to the ImageNet ResNets for CIFAR-scale benchmarking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper

_STAGE_CHANNELS = (16, 32, 64)


class _BasicBlock(nn.Module):
    """Pre-activation-free basic block (matches He et al. 2015 figure 5).

    Two 3x3 convs with BN+ReLU; first conv handles the stride for the
    downsampling block at the start of stages 2 and 3.
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):
    """CIFAR-native ResNet backbone with stage1/stage2/stage3 submodules.

    Submodule names ``stage1``, ``stage2``, ``stage3`` are resolvable via
    ``get_submodule`` so they can be used as early-exit attachment points.
    """

    def __init__(self, n: int, num_classes: int) -> None:
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, _STAGE_CHANNELS[0], n, stride=1)
        self.stage2 = self._make_stage(_STAGE_CHANNELS[0], _STAGE_CHANNELS[1], n, stride=2)
        self.stage3 = self._make_stage(_STAGE_CHANNELS[1], _STAGE_CHANNELS[2], n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(_STAGE_CHANNELS[2], num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [_BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(_BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def cifar_resnet_ee(
    num_classes: int,
    depth: int = 56,
    pretrained: bool = False,
) -> EarlyExitWrapper:
    """Build a CIFAR-native ResNet wrapped with three early exits.

    Parameters
    ----------
    num_classes:
        Output class count (10 for CIFAR-10, 100 for CIFAR-100).
    depth:
        Total layer count. Must satisfy ``depth == 6 * n + 2`` for some
        positive integer ``n`` (per He et al. 2015 §4.2). Valid values:
        8, 14, 20, 26, 32, 38, 44, 50, 56, 110, ...
    pretrained:
        Ignored — no public pretrained CIFAR-ResNet weights ship with
        torchvision. Accepted for signature symmetry with the ImageNet
        factories.
    """
    if (depth - 2) <= 0 or (depth - 2) % 6 != 0:
        raise ValueError(
            f"depth must be 6n+2 with n>=1 (got {depth}); valid examples: "
            "8, 14, 20, 26, 32, 38, 44, 50, 56, 110"
        )
    n = (depth - 2) // 6
    backbone = CifarResNet(n=n, num_classes=num_classes)
    if pretrained:
        # accept silently; surfacing a warning would noise normal usage
        pass

    exit_points = [
        ExitPoint("e0", "stage1", _STAGE_CHANNELS[0]),
        ExitPoint("e1", "stage2", _STAGE_CHANNELS[1]),
        ExitPoint("e2", "stage3", _STAGE_CHANNELS[2]),
    ]
    heads = {
        ep.name: EarlyExitHead(ep.in_channels, num_classes, hidden_dim=64)
        for ep in exit_points
    }
    cfg = EarlyExitConfig(
        backbone=f"cifar_resnet{depth}",
        num_classes=num_classes,
        exit_points=exit_points,
        confidence_thresholds=[0.9, 0.85, 0.8],
        loss_weights=[0.15, 0.2, 0.25, 0.4],
    )
    return EarlyExitWrapper(backbone, heads, _identity, cfg, input_shape=(1, 3, 32, 32))
