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

import math
import warnings
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from earlyon.core.temperature import fit_temperature as _fit_temperature
from earlyon.core.types import Batch
from earlyon.core.wrappers import EarlyExitWrapper


@dataclass
class CalibrationResult:
    thresholds: list[float]
    baseline_accuracy: float
    final_accuracy: float
    avg_computation_used: float
    fitted_temperature: float | None = None
    policy: str = "confidence"  # which routing policy the thresholds calibrate


# confidence grid: ordered conservative -> aggressive (lower threshold fires more)
DEFAULT_GRID: tuple[float, ...] = (0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)

# entropy grid: fractions of H_max = ln(num_classes), ordered conservative ->
# aggressive (higher entropy threshold fires more, since exit fires on H <= thr)
DEFAULT_ENTROPY_FRACTIONS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _evaluate(
    model: EarlyExitWrapper, loader: DataLoader[Batch], device: str
) -> tuple[float, float]:
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
    model: EarlyExitWrapper, loader: DataLoader[Batch], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``loader`` through the wrapper in training mode and harvest the
    final classifier's logits + targets. Training mode never fires an exit (the
    hook only appends logits and never raises), so ``outputs[-1]`` is always the
    final classifier and no threshold reset is needed."""
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
    val_loader: DataLoader[Batch],
    target_accuracy_drop: float = 0.01,
    grid: tuple[float, ...] | None = None,
    device: str = "cpu",
    fit_temperature: bool = False,
    temperature_loader: DataLoader[Batch] | None = None,
) -> CalibrationResult:
    """Find the most aggressive thresholds that keep accuracy within target.

    Calibration follows ``model.config.routing_policy``:

    * ``"confidence"`` — calibrates ``confidence_thresholds``; the router exits
      when ``softmax.max() >= threshold``, so a *lower* threshold is more
      aggressive. Default grid is :data:`DEFAULT_GRID` (probabilities).
    * ``"entropy"`` — calibrates ``entropy_thresholds``; the router exits when
      ``H(softmax) <= threshold``, so a *higher* threshold is more aggressive.
      Default grid is :data:`DEFAULT_ENTROPY_FRACTIONS` scaled by
      ``ln(num_classes)``.

    The result is written to whichever threshold list the active policy reads —
    calibrating an entropy-routed model under the old confidence-only code was a
    silent no-op.

    Parameters
    ----------
    grid:
        Optional custom search grid. If ``None`` a policy-appropriate default is
        used. For entropy a custom grid must lie in ``[0, ln(num_classes)]``.
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
    policy = model.config.routing_policy

    # Per-policy search setup. ``seed`` is a strictly non-firing value used while
    # measuring the baseline and untried exits — it must NEVER fire even on a
    # float32-saturated softmax (max prob == 1.0 / entropy == -0.0), or the
    # no-exit baseline would be corrupted. ``no_exit`` is the in-range "disabled"
    # value the seed is clamped back to in the result. ``more_aggressive`` decides
    # which passing threshold to keep (smaller for confidence, larger for entropy).
    if policy == "confidence":
        field = "confidence_thresholds"
        seed = 2.0  # softmax.max() <= 1.0, so confidence >= 2.0 can never fire
        no_exit = 1.0  # documented "no early exit at this point" value
        search_grid = DEFAULT_GRID if grid is None else grid

        def more_aggressive(new: float, cur: float) -> bool:
            return new < cur

    elif policy == "entropy":
        field = "entropy_thresholds"
        seed = -1.0  # entropy >= 0, so H <= -1.0 can never fire (even at -0.0)
        no_exit = 0.0
        h_max = math.log(max(model.config.num_classes, 2))
        if grid is None:
            search_grid = tuple(round(f * h_max, 6) for f in DEFAULT_ENTROPY_FRACTIONS)
        else:
            if any(t < 0 or t > h_max + 1e-9 for t in grid):
                raise ValueError(
                    f"entropy grid values must lie in [0, ln(num_classes)] = "
                    f"[0, {h_max:.4f}]; got {grid}"
                )
            search_grid = grid

        def more_aggressive(new: float, cur: float) -> bool:
            return new > cur

    else:  # pragma: no cover - EarlyExitConfig already validates the policy
        raise ValueError(f"unsupported routing_policy {policy!r}")

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

        # Training-mode collection never fires exits (the hook only appends in
        # training mode), so thresholds are irrelevant here; only the
        # temperature must be neutral while harvesting uncalibrated logits.
        original_temp = model.config.temperature
        model.config.temperature = 1.0
        try:
            logits, targets = _collect_final_logits(model, temperature_loader, device)
        finally:
            model.config.temperature = original_temp
        fitted_t = _fit_temperature(logits.cpu(), targets.cpu())
        model.config.temperature = fitted_t

    # start fully conservative -- no exits will fire
    best = [seed] * n
    setattr(model.config, field, list(best))
    baseline_acc, _ = _evaluate(model, val_loader, device)

    # iterate the full grid per exit. an early break would miss thresholds
    # that pass after a transient miss (val accuracy isn't monotone in threshold
    # on small sets). keep the most aggressive passing threshold seen.
    for exit_idx in range(n):
        for thr in search_grid:
            trial = list(best)
            trial[exit_idx] = thr
            setattr(model.config, field, list(trial))
            acc, _ = _evaluate(model, val_loader, device)
            if baseline_acc - acc <= target_accuracy_drop and more_aggressive(thr, best[exit_idx]):
                best[exit_idx] = thr

    # clamp any exit that never found a passing threshold from the non-firing
    # search seed back to the in-range disabled value.
    best = [no_exit if t == seed else t for t in best]
    setattr(model.config, field, list(best))
    final_acc, avg_comp = _evaluate(model, val_loader, device)
    return CalibrationResult(
        thresholds=best,
        baseline_accuracy=baseline_acc,
        final_accuracy=final_acc,
        avg_computation_used=avg_comp,
        fitted_temperature=fitted_t,
        policy=policy,
    )
