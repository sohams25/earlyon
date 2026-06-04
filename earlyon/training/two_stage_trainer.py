"""Two-stage trainer: stage 1 trains backbone standalone, stage 2 freezes
backbone and trains exit heads only. This is the strategy recommended by
the ACM 2024 early-exit survey: simpler, faster, and avoids gradient
conflict between exits and backbone.

CRITICAL: during stage 2 we ``backbone.eval()`` and keep it there. Setting
``requires_grad=False`` does NOT disable BatchNorm running-stat updates;
forgetting this causes exit-head accuracy to drift between runs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training.losses import weighted_multi_exit_loss

_Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TrainStepLog:
    epoch: int
    loss: float
    accuracy: float  # stage1: backbone accuracy; stage2: mean across all outputs
    per_exit_accuracy: dict[str, float] | None = None  # populated in stage 2 only


LogFn = Callable[[TrainStepLog], None]


def _warn_if_val_loader_unused(val_loader: DataLoader[_Batch] | None) -> None:
    """The trainers accept ``val_loader`` for forward-compatibility but do not
    yet use it (no early stopping or best-checkpoint selection). Warn rather
    than silently ignore so callers don't assume validation-driven behavior."""
    if val_loader is not None:
        warnings.warn(
            "val_loader is accepted for forward-compatibility but is not yet "
            "used: no early stopping or best-checkpoint selection is performed. "
            "Evaluate on the validation set separately after training.",
            UserWarning,
            stacklevel=3,
        )


def _default_log(log: TrainStepLog) -> None:
    # print() is intentional: this is the default callback for the CLI,
    # not internal library logging.
    base = f"epoch {log.epoch}: loss={log.loss:.4f} acc={log.accuracy:.4f}"
    if log.per_exit_accuracy:
        parts = " ".join(f"{k}={v:.4f}" for k, v in log.per_exit_accuracy.items())
        base = f"{base} ({parts})"
    print(base)


def stage1_train_backbone(
    backbone: nn.Module,
    train_loader: DataLoader[_Batch],
    val_loader: DataLoader[_Batch] | None = None,
    epochs: int = 90,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
) -> nn.Module:
    """Train ``backbone`` as a standard classifier. Exits are not involved.

    ``val_loader`` is accepted but not yet used (see
    :func:`_warn_if_val_loader_unused`).
    """
    _warn_if_val_loader_unused(val_loader)
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
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
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
    train_loader: DataLoader[_Batch],
    val_loader: DataLoader[_Batch] | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
) -> EarlyExitWrapper:
    """Freeze backbone, train only exit heads with weighted multi-exit CE.

    ``val_loader`` is accepted but not yet used (see
    :func:`_warn_if_val_loader_unused`).
    """
    _warn_if_val_loader_unused(val_loader)
    model = model.to(device)

    # freeze backbone parameters AND keep BatchNorm in eval mode
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # only exit heads train
    for param in model.exit_parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.exit_parameters(), lr=lr, weight_decay=weight_decay)

    n_exits = len(model.config.exit_points)
    output_labels = [f"exit_{i}" for i in range(n_exits)] + ["final"]

    for epoch in range(epochs):
        # exit heads in train mode; backbone stays in eval
        for head in model.exit_heads.values():
            head.train()
        model.backbone.eval()

        running_loss = 0.0
        # per-output correct counts: index 0..n_exits-1 are exits, last is final
        correct_per_output = [0] * (n_exits + 1)
        total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images, mode="training")
            loss = weighted_multi_exit_loss(outputs, targets, model.config.loss_weights)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            for i, out in enumerate(outputs):
                correct_per_output[i] += (out.argmax(dim=-1) == targets).sum().item()
            total += images.size(0)

        per_exit_acc = {
            label: correct_per_output[i] / max(total, 1) for i, label in enumerate(output_labels)
        }
        mean_acc = sum(per_exit_acc.values()) / len(per_exit_acc)
        on_epoch_end(
            TrainStepLog(
                epoch=epoch,
                loss=running_loss / max(total, 1),
                accuracy=mean_acc,
                per_exit_accuracy=per_exit_acc,
            )
        )
    return model
