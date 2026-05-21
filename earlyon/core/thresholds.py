"""Greedy threshold calibration.

For each exit (low-index to high-index), lower the confidence threshold
through a fixed grid until the resulting accuracy drop on the validation set
exceeds ``target_accuracy_drop``. The search is coordinate-descent, not
joint-optimal; this is fast and gives consistently good results in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from earlyon.core.wrappers import EarlyExitWrapper


@dataclass
class CalibrationResult:
    thresholds: list[float]
    baseline_accuracy: float
    final_accuracy: float
    avg_computation_used: float


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


def calibrate_thresholds(
    model: EarlyExitWrapper,
    val_loader: DataLoader,
    target_accuracy_drop: float = 0.01,
    grid: tuple[float, ...] = DEFAULT_GRID,
    device: str = "cpu",
) -> CalibrationResult:
    """Find the most aggressive thresholds that keep accuracy within target."""
    model = model.to(device)
    n = len(model.config.exit_points)

    # start fully conservative -- no exits will fire
    original = list(model.config.confidence_thresholds)
    best = [1.0] * n
    model.config.confidence_thresholds = list(best)
    baseline_acc, _ = _evaluate(model, val_loader, device)

    for exit_idx in range(n):
        for thr in grid:
            trial = list(best)
            trial[exit_idx] = thr
            model.config.confidence_thresholds = list(trial)
            acc, _ = _evaluate(model, val_loader, device)
            if baseline_acc - acc <= target_accuracy_drop:
                best[exit_idx] = thr
            else:
                break

    model.config.confidence_thresholds = list(best)
    final_acc, avg_comp = _evaluate(model, val_loader, device)
    # restore-or-keep: we keep the calibrated thresholds set on the model,
    # but return original for reference if needed (not used currently)
    _ = original
    return CalibrationResult(
        thresholds=best,
        baseline_accuracy=baseline_acc,
        final_accuracy=final_acc,
        avg_computation_used=avg_comp,
    )
