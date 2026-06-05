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

from earlyon.core.types import Batch, exit_label
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training.losses import weighted_multi_exit_loss


@dataclass
class TrainStepLog:
    epoch: int
    loss: float
    accuracy: float  # stage1: backbone accuracy; stage2: mean across all outputs
    per_exit_accuracy: dict[str, float] | None = None  # populated in stage 2 only
    val_loss: float | None = None  # set when a val_loader is supplied
    val_accuracy: float | None = None


LogFn = Callable[[TrainStepLog], None]


def _default_log(log: TrainStepLog) -> None:
    # print() is intentional: this is the default callback for the CLI,
    # not internal library logging.
    base = f"epoch {log.epoch}: loss={log.loss:.4f} acc={log.accuracy:.4f}"
    if log.per_exit_accuracy:
        parts = " ".join(f"{k}={v:.4f}" for k, v in log.per_exit_accuracy.items())
        base = f"{base} ({parts})"
    if log.val_loss is not None and log.val_accuracy is not None:
        base = f"{base} val_loss={log.val_loss:.4f} val_acc={log.val_accuracy:.4f}"
    print(base)


def _validate_backbone(
    backbone: nn.Module, loader: DataLoader[Batch], device: str
) -> tuple[float, float]:
    """Mean cross-entropy loss and top-1 accuracy of the backbone on ``loader``.

    Runs in eval/no_grad and restores the caller's train/eval mode on exit, so a
    val_loader never changes the returned model's mode (mirrors per_layer_flops).
    """
    was_training = backbone.training
    backbone.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = backbone(images)
                loss_sum += F.cross_entropy(logits, targets).item() * images.size(0)
                correct += int((logits.argmax(dim=-1) == targets).sum().item())
                total += images.size(0)
    finally:
        backbone.train(was_training)
    if total == 0:
        raise ValueError("val_loader yielded no batches")
    return loss_sum / total, correct / total


def _validate_wrapper(
    model: EarlyExitWrapper, loader: DataLoader[Batch], device: str
) -> tuple[float, float]:
    """Weighted multi-exit loss and mean accuracy across all exits + final.

    Uses ``mode="training"`` (every head produces logits) under eval/no_grad, so
    BatchNorm/dropout are deterministic and no exit short-circuits the pass. The
    per-submodule train/eval modes are restored on exit so a val_loader never
    changes the returned model's mode.
    """
    saved_modes = {name: m.training for name, m in model.named_modules()}
    model.eval()
    weights = model.config.loss_weights
    loss_sum = 0.0
    correct = 0
    total_outputs = 0
    total_samples = 0
    try:
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images, mode="training")
                loss_sum += weighted_multi_exit_loss(
                    outputs, targets, weights
                ).item() * images.size(0)
                for out in outputs:
                    correct += int((out.argmax(dim=-1) == targets).sum().item())
                    # one prediction per (sample, head): total_outputs == K * samples,
                    # so correct / total_outputs is the mean per-head top-1 accuracy.
                    total_outputs += targets.size(0)
                total_samples += images.size(0)
    finally:
        for name, module in model.named_modules():
            module.train(saved_modes[name])
    if total_samples == 0:
        raise ValueError("val_loader yielded no batches")
    return loss_sum / total_samples, correct / total_outputs


def stage1_train_backbone(
    backbone: nn.Module,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch] | None = None,
    epochs: int = 90,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
) -> nn.Module:
    """Train ``backbone`` as a standard classifier. Exits are not involved.

    When ``val_loader`` is given, validation loss/accuracy are computed at the
    end of each epoch and reported via ``TrainStepLog.val_loss``/``val_accuracy``.
    """
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

        val_loss, val_acc = (None, None)
        if val_loader is not None:
            val_loss, val_acc = _validate_backbone(backbone, val_loader, device)
        on_epoch_end(
            TrainStepLog(
                epoch=epoch,
                loss=running_loss / max(total, 1),
                accuracy=correct / max(total, 1),
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )
    return backbone


def stage2_train_exits(
    model: EarlyExitWrapper,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch] | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
) -> EarlyExitWrapper:
    """Freeze backbone, train only exit heads with weighted multi-exit CE.

    When ``val_loader`` is given, validation metrics are reported each epoch.
    """
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
    output_labels = [exit_label(i) for i in range(n_exits)] + [exit_label(-1)]

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

        val_loss, val_acc = (None, None)
        if val_loader is not None:
            val_loss, val_acc = _validate_wrapper(model, val_loader, device)
        on_epoch_end(
            TrainStepLog(
                epoch=epoch,
                loss=running_loss / max(total, 1),
                accuracy=mean_acc,
                per_exit_accuracy=per_exit_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )
    return model
