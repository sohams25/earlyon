"""Loss functions for multi-exit training."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def weighted_multi_exit_loss(
    predictions: Sequence[torch.Tensor],
    targets: torch.Tensor,
    weights: Sequence[float],
) -> torch.Tensor:
    """Sum of per-exit cross-entropy, scaled by per-exit weights.

    ``predictions`` contains one logits tensor per exit head plus one for the
    final classifier (so ``len(predictions) == len(weights)``).
    """
    if len(predictions) != len(weights):
        raise ValueError(
            f"got {len(predictions)} predictions but {len(weights)} weights"
        )
    loss = predictions[0].new_zeros(())
    for pred, w in zip(predictions, weights):
        loss = loss + w * F.cross_entropy(pred, targets)
    return loss
