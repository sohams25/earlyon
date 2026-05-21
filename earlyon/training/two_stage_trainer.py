"""Two-stage trainer: stage 1 trains backbone standalone, stage 2 freezes
backbone and trains exit heads only. This is the strategy recommended by
the ACM 2024 early-exit survey: simpler, faster, and avoids gradient
conflict between exits and backbone.

CRITICAL: during stage 2 we ``backbone.eval()`` and keep it there. Setting
``requires_grad=False`` does NOT disable BatchNorm running-stat updates;
forgetting this causes exit-head accuracy to drift between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training.losses import weighted_multi_exit_loss


@dataclass
class TrainStepLog:
    epoch: int
    loss: float
    accuracy: float


LogFn = Callable[[TrainStepLog], None]


def _default_log(log: TrainStepLog) -> None:
    print(f"epoch {log.epoch}: loss={log.loss:.4f} acc={log.accuracy:.4f}")


def stage1_train_backbone(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 90,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
) -> nn.Module:
    """Train ``backbone`` as a standard classifier. Exits are not involved."""
    backbone = backbone.to(device)
    optimizer = torch.optim.SGD(
        backbone.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        backbone.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = backbone(images)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += images.size(0)
        scheduler.step()
        on_epoch_end(
            TrainStepLog(
                epoch=epoch,
                loss=running_loss / max(total, 1),
                accuracy=correct / max(total, 1),
            )
        )
    return backbone


def stage2_train_exits(
    model: EarlyExitWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
) -> EarlyExitWrapper:
    """Freeze backbone, train only exit heads with weighted multi-exit CE."""
    model = model.to(device)

    # freeze backbone parameters AND keep BatchNorm in eval mode
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # only exit heads train
    for param in model.exit_parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.exit_parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        # exit heads in train mode; backbone stays in eval
        for head in model.exit_heads.values():
            head.train()
        model.backbone.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images, mode="training")
            loss = weighted_multi_exit_loss(outputs, targets, model.config.loss_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # report accuracy from the first exit (most aggressive)
            running_loss += loss.item() * images.size(0)
            correct += (outputs[0].argmax(dim=-1) == targets).sum().item()
            total += images.size(0)
        on_epoch_end(
            TrainStepLog(
                epoch=epoch,
                loss=running_loss / max(total, 1),
                accuracy=correct / max(total, 1),
            )
        )
    return model
