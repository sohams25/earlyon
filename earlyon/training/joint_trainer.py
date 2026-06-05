"""Joint trainer: train backbone and exit heads together in one loop.

This is the alternative to ``two_stage_trainer``. Both run; the README and
literature still recommend the two-stage recipe as the default — it avoids
gradient conflict between the exits and the backbone — but joint training
is the path users want when they care about absolute peak accuracy and have
the compute budget to converge end-to-end.

Difference from stage 2:
* backbone parameters are NOT frozen (``requires_grad=True``)
* backbone stays in ``train()`` mode so BatchNorm running stats update
* a single optimizer steps every parameter in the wrapper
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from earlyon.core.types import Batch, exit_label
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training.losses import weighted_multi_exit_loss
from earlyon.training.two_stage_trainer import (
    LogFn,
    TrainStepLog,
    _default_log,
    _validate_wrapper,
)


def joint_train_backbone_and_exits(
    model: EarlyExitWrapper,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch] | None = None,
    epochs: int = 30,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    on_epoch_end: LogFn = _default_log,
    optimizer: torch.optim.Optimizer | None = None,
) -> EarlyExitWrapper:
    """Train the wrapper end-to-end with a single weighted-multi-exit loss.

    Pass ``optimizer`` to override the default SGD; the default takes every
    parameter of ``model`` (backbone + exit heads) so callers don't accidentally
    freeze the backbone the way two-stage training does.

    When ``val_loader`` is given, validation loss/accuracy are reported each
    epoch via ``TrainStepLog.val_loss``/``val_accuracy``.
    """
    model = model.to(device)

    # explicitly enable grads on backbone — defensive: users who previously
    # ran stage 2 may have left requires_grad=False on the backbone
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.exit_parameters():
        param.requires_grad = True

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_exits = len(model.config.exit_points)
    output_labels = [exit_label(i) for i in range(n_exits)] + [exit_label(-1)]

    for epoch in range(epochs):
        # backbone + exits all in train mode (BN running stats update)
        model.backbone.train()
        for head in model.exit_heads.values():
            head.train()

        running_loss = 0.0
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

        scheduler.step()

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
