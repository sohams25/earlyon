"""Tiny CNN backbones for fast tests. Real architectures live in earlyon.models."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyBackbone(nn.Module):
    """4-stage toy CNN: 3 -> 16 -> 32 -> 64 -> 128 channels.

    Each stage is one conv + ReLU + maxpool. Final stage is global-pooled and
    passed through ``self.fc``. Layer names ``stage1..stage4`` are valid
    targets for early-exit hooks.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stage1 = self._stage(3, 16)
        self.stage2 = self._stage(16, 32)
        self.stage3 = self._stage(32, 64)
        self.stage4 = self._stage(64, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    @staticmethod
    def _stage(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


STAGE_CHANNELS = {"stage1": 16, "stage2": 32, "stage3": 64, "stage4": 128}


class TinyTokenBackbone(nn.Module):
    """Tiny transformer-style backbone whose ``block0``/``block1`` emit token
    sequences ``(B, N, D)`` — used to exercise the 3D early-exit path without the
    cost of a real ViT. ``forward`` returns ``(B, num_classes)`` logits.
    """

    def __init__(self, num_classes: int = 10, dim: int = 32, n_tokens: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.n_tokens = n_tokens
        self.embed = nn.Linear(3 * 32 * 32, dim)
        self.block0 = nn.TransformerEncoderLayer(dim, nhead=4, batch_first=True)
        self.block1 = nn.TransformerEncoderLayer(dim, nhead=4, batch_first=True)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x.flatten(1)).unsqueeze(1).repeat(1, self.n_tokens, 1)
        tokens = self.block1(self.block0(tokens))
        return self.head(tokens.mean(dim=1))
