"""Greedy threshold calibration.

For each exit (low-index to high-index), lower the confidence threshold
through a fixed grid until the resulting accuracy drop on the validation set
exceeds ``target_accuracy_drop``. The search is coordinate-descent, not
joint-optimal; this is fast and gives consistently good results in practice.

When ``fit_temperature=True``, post-hoc temperature scaling (Guo et al. 2017)
is fit on a held-out set BEFORE the threshold search begins. The fitted
scalar lands on ``model.config.temperature`` so every subsequent softmax
(inside the hook and at the final classifier) is calibrated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from earlyon.core.temperature import fit_temperature as _fit_temperature
from earlyon.core.wrappers import EarlyExitWrapper


@dataclass
class CalibrationResult:
    thresholds: list[float]
    baseline_accuracy: float
    final_accuracy: float
    avg_computation_used: float
    fitted_temperature: float | None = None


DEFAULT_GRID: tuple[float, ...] = (0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)


def _evaluate(model: EarlyExitWrapper, loader: DataLoader, device: str) -> tuple[float, float]:
    correct = 0
    total = 0
    comp = 0.0
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            for img, tgt in zip(images, targets):
                result = model(img.unsqueeze(0), mode="inference")
                pred = int(result.prediction.argmax(dim=-1).item())
                correct += int(pred == int(tgt.item()))
                comp += result.computation_used
                total += 1
    return correct / max(total, 1), comp / max(total, 1)


def _collect_final_logits(
    model: EarlyExitWrapper, loader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``loader`` through the wrapper in training mode and harvest the
    final classifier's logits + targets. Hook-fired exits would short-circuit
    routing, so the caller must set thresholds to 1.0 first."""
    model = model.to(device)
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images, mode="training")
            # outputs is [exit_0_logits, ..., exit_n_logits, final_logits]
            all_logits.append(outputs[-1].detach())
            all_targets.append(targets.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def calibrate_thresholds(
    model: EarlyExitWrapper,
    val_loader: DataLoader,
    target_accuracy_drop: float = 0.01,
    grid: tuple[float, ...] = DEFAULT_GRID,
    device: str = "cpu",
    fit_temperature: bool = False,
    temperature_loader: DataLoader | None = None,
) -> CalibrationResult:
    """Find the most aggressive thresholds that keep accuracy within target.

    Parameters
    ----------
    fit_temperature:
        If True, fit post-hoc temperature scaling before the threshold search.
        The fitted scalar is written to ``model.config.temperature``.
    temperature_loader:
        DataLoader used to fit the temperature. If ``None`` and
        ``fit_temperature`` is True, ``val_loader`` is reused — this leaks the
        temperature fit into the threshold accuracy estimate, so a warning
        is emitted. Prefer a separate held-out split when available.
    """
    model = model.to(device)
    n = len(model.config.exit_points)

    fitted_t: float | None = None
    if fit_temperature:
        if temperature_loader is None:
            warnings.warn(
                "fit_temperature=True without a separate temperature_loader: "
                "val_loader is being reused for both temperature fit and "
                "threshold search, which leaks the calibration estimate. Pass "
                "a held-out temperature_loader for a clean fit.",
                UserWarning,
                stacklevel=2,
            )
            temperature_loader = val_loader

        # disable routing for collection (no exit fires)
        original_thresholds = list(model.config.confidence_thresholds)
        model.config.confidence_thresholds = [1.0] * n
        model.config.temperature = 1.0  # collect uncalibrated logits
        try:
            logits, targets = _collect_final_logits(model, temperature_loader, device)
        finally:
            model.config.confidence_thresholds = original_thresholds
        fitted_t = _fit_temperature(logits.cpu(), targets.cpu())
        model.config.temperature = fitted_t

    # start fully conservative -- no exits will fire
    best = [1.0] * n
    model.config.confidence_thresholds = list(best)
    baseline_acc, _ = _evaluate(model, val_loader, device)

    # iterate the full grid per exit. an early break would miss thresholds
    # that pass after a transient miss (val accuracy isn't monotone in threshold
    # on small sets). keep the lowest passing threshold seen.
    for exit_idx in range(n):
        for thr in grid:
            trial = list(best)
            trial[exit_idx] = thr
            model.config.confidence_thresholds = list(trial)
            acc, _ = _evaluate(model, val_loader, device)
            if baseline_acc - acc <= target_accuracy_drop:
                # the grid is ordered high->low so the most recently passing
                # threshold is the lowest one that satisfies the constraint
                if thr < best[exit_idx]:
                    best[exit_idx] = thr

    model.config.confidence_thresholds = list(best)
    final_acc, avg_comp = _evaluate(model, val_loader, device)
    return CalibrationResult(
        thresholds=best,
        baseline_accuracy=baseline_acc,
        final_accuracy=final_acc,
        avg_computation_used=avg_comp,
        fitted_temperature=fitted_t,
    )
