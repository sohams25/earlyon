"""Lightweight classifier head attached at intermediate backbone layers."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyExitHead(nn.Module):
    """Pool -> linear -> relu -> dropout -> linear.

    Accepts three feature shapes (pass the matching ``in_channels``):

    * 4D ``(B, C, H, W)`` — conv maps: global average pool over H,W, then
      classify on ``C``.
    * 3D ``(B, N, D)`` — transformer tokens: pooled to ``(B, D)`` per
      ``pool_tokens``, then classify on ``D``.
    * 2D ``(B, D)`` — already-pooled vectors: passed through unchanged.

    ``pool_tokens`` controls 3D pooling: ``"cls"`` takes token 0 (mirrors the
    ViT readout); ``"mean"`` averages over the token dimension
    (architecture-agnostic — the default ``custom_ee`` uses for unknown backbones).

    Designed to be small: 512 channels -> 10 classes is ~70K params, negligible
    next to a 2048-channel ResNet50 FC.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pool_tokens: str = "cls",
    ) -> None:
        super().__init__()
        if pool_tokens not in ("cls", "mean"):
            raise ValueError(f"pool_tokens must be 'cls' or 'mean', got {pool_tokens!r}")
        self.pool_tokens = pool_tokens
        # No stored pool module: F.adaptive_avg_pool2d handles the 4D path and
        # AdaptiveAvgPool2d registered no state_dict keys anyway.
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = self.flatten(F.adaptive_avg_pool2d(x, 1))
        elif x.dim() == 3:
            x = x[:, 0, :] if self.pool_tokens == "cls" else x.mean(dim=1)
        elif x.dim() != 2:
            raise ValueError(
                f"EarlyExitHead expects 2D, 3D, or 4D input; got {x.dim()}D "
                f"tensor of shape {tuple(x.shape)}"
            )
        return cast(torch.Tensor, self.classifier(x))
