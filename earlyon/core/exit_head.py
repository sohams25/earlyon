"""Lightweight classifier head attached at intermediate backbone layers."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class EarlyExitHead(nn.Module):
    """Pool -> linear -> relu -> dropout -> linear.

    Designed to be small: for 512-channel features and 10 classes, this is
    ~70K params, which is negligible vs. a 2048-channel ResNet50 FC.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        # nn.Sequential.__call__ is typed -> Any in the torch stubs; the runtime
        # value is always a Tensor.
        return cast(torch.Tensor, self.classifier(x))
