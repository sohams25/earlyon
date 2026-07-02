"""Greedy threshold calibration.

Two calibration modes share one greedy coordinate-descent search:

* :func:`calibrate_thresholds` — accuracy budget. For each exit (low-index to
  high-index), lower the threshold through a fixed grid until the resulting
  accuracy drop on the validation set exceeds ``target_accuracy_drop``.
* :func:`calibrate_thresholds_for_budget` — compute budget. For each exit,
  keep the threshold that brings average compute within ``target_computation``
  while losing the least accuracy; exits that cannot help stay disabled.

The search is coordinate-descent, not joint-optimal; this is fast and gives
consistently good results in practice. Trials are evaluated against
:class:`_EvalCache` — one batched forward pass collects every head's logits,
and each threshold setting is then simulated vectorially, so the network runs
once instead of once per grid point.

When ``fit_temperature=True``, post-hoc temperature scaling (Guo et al. 2017)
is fit on a held-out set BEFORE the threshold search begins. The fitted
scalar lands on ``model.config.temperature`` so every subsequent softmax
(inside the hook and at the final classifier) is calibrated.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader

from earlyon.core.temperature import fit_temperature as _fit_temperature
from earlyon.core.types import Batch
from earlyon.core.wrappers import EarlyExitWrapper, _safe_temperature


@dataclass
class CalibrationResult:
    thresholds: list[float]
    baseline_accuracy: float
    final_accuracy: float
    avg_computation_used: float
    fitted_temperature: float | None = None
    policy: str = "confidence"  # which routing policy the thresholds calibrate
    target_computation: float | None = None  # set by budget calibration
    budget_met: bool = True  # False when target_computation was unattainable


# confidence grid: ordered conservative -> aggressive (lower threshold fires more)
DEFAULT_GRID: tuple[float, ...] = (0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)

# entropy grid: fractions of H_max = ln(num_classes), ordered conservative ->
# aggressive (higher entropy threshold fires more, since exit fires on H <= thr)
DEFAULT_ENTROPY_FRACTIONS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _evaluate_with_router(
    model: EarlyExitWrapper, loader: DataLoader[Batch], device: str
) -> tuple[float, float]:
    """Ground truth: route every sample through the real inference path.

    Kept as the oracle that pins :class:`_EvalCache` correctness in the test
    suite; calibration itself uses the cache."""
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


class _EvalCache:
    """One batched training-mode pass caches every head's logits; threshold
    trials are then simulated against the cache instead of re-running the
    network.

    The simulation reproduces the router's math exactly: softmax in float32
    (as the hook computes it), criterion comparison in float64 (as the hook's
    ``.item()`` produces), first firing exit claims the sample. A test pins
    equivalence against :func:`_evaluate_with_router`.

    Memory: stores ``(n_exits + 1) x N x num_classes`` logits on CPU, which is
    fine for calibration-sized validation sets.

    ponytail: turns grid search from O(exits * grid) full validation passes at
    batch size 1 into one batched pass plus cheap tensor math.
    """

    def __init__(self, model: EarlyExitWrapper, loader: DataLoader[Batch], device: str) -> None:
        self._model = model
        per_head: list[list[torch.Tensor]] = []
        targets: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for images, tgt in loader:
                outputs = model(images.to(device), mode="training")
                if not per_head:
                    per_head = [[] for _ in outputs]
                for idx, logits in enumerate(outputs):
                    per_head[idx].append(logits.detach().cpu())
                targets.append(tgt.cpu())
        if targets:
            self._heads = [torch.cat(chunks) for chunks in per_head]
            self._targets = torch.cat(targets)
        else:  # empty loader: preserve the old 0.0/0.0 contract
            self._heads = []
            self._targets = torch.empty(0, dtype=torch.long)
        exit_flops = [model._flops_at[ep.layer_name] for ep in model.config.exit_points]
        self._flops = torch.tensor(exit_flops + [1.0], dtype=torch.float64)

    def evaluate(self) -> tuple[float, float]:
        """(accuracy, avg computation used) under the model's current config."""
        if self._targets.numel() == 0:
            return 0.0, 0.0
        cfg = self._model.config
        temp = _safe_temperature(cfg.temperature)
        n_exits = len(cfg.exit_points)
        n = self._targets.shape[0]
        chosen = torch.full((n,), n_exits, dtype=torch.long)
        for idx in range(n_exits - 1, -1, -1):  # earlier exits override later ones
            probs = torch.softmax(self._heads[idx] / temp, dim=-1)
            if cfg.routing_policy == "confidence":
                confidence = probs.max(dim=-1).values.double()
                fires = confidence >= cfg.confidence_thresholds[idx]
            else:
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                fires = entropy.double() <= cfg.entropy_thresholds[idx]
            chosen[fires] = idx
        preds = torch.stack([h.argmax(dim=-1) for h in self._heads])
        pred = preds[chosen, torch.arange(n)]
        acc = (pred == self._targets).double().mean().item()
        comp = self._flops[chosen].mean().item()
        return acc, comp


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


@dataclass(frozen=True)
class _PolicySearch:
    """Per-policy search setup. ``seed`` is a strictly non-firing value used
    while measuring the baseline and untried exits — it must NEVER fire even on
    a float32-saturated softmax (max prob == 1.0 / entropy == -0.0), or the
    no-exit baseline would be corrupted. ``no_exit`` is the in-range "disabled"
    value the seed is clamped back to in the result. ``more_aggressive`` decides
    which passing threshold to keep (smaller for confidence, larger for
    entropy)."""

    field: str
    seed: float
    no_exit: float
    grid: tuple[float, ...]
    more_aggressive: Callable[[float, float], bool]


def _policy_search(model: EarlyExitWrapper, grid: tuple[float, ...] | None) -> _PolicySearch:
    if grid is not None and len(grid) == 0:
        # an empty grid searches nothing: calibration would "succeed" with all
        # exits disabled and the model would never exit early
        raise ValueError("custom grid must be non-empty (or None for the default)")
    policy = model.config.routing_policy
    if policy == "confidence":
        return _PolicySearch(
            field="confidence_thresholds",
            seed=2.0,  # softmax.max() <= 1.0, so confidence >= 2.0 can never fire
            no_exit=1.0,  # documented "no early exit at this point" value
            grid=DEFAULT_GRID if grid is None else grid,
            more_aggressive=lambda new, cur: new < cur,
        )
    if policy == "entropy":
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
        return _PolicySearch(
            field="entropy_thresholds",
            seed=-1.0,  # entropy >= 0, so H <= -1.0 can never fire (even at -0.0)
            no_exit=0.0,
            grid=search_grid,
            more_aggressive=lambda new, cur: new > cur,
        )
    raise ValueError(  # pragma: no cover - EarlyExitConfig already validates the policy
        f"unsupported routing_policy {policy!r}"
    )


def _maybe_fit_temperature(
    model: EarlyExitWrapper,
    fit_temperature: bool,
    temperature_loader: DataLoader[Batch] | None,
    val_loader: DataLoader[Batch],
    device: str,
) -> float | None:
    if not fit_temperature:
        return None
    if temperature_loader is None:
        warnings.warn(
            "fit_temperature=True without a separate temperature_loader: "
            "val_loader is being reused for both temperature fit and "
            "threshold search, which leaks the calibration estimate. Pass "
            "a held-out temperature_loader for a clean fit.",
            UserWarning,
            stacklevel=3,
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
    fitted = _fit_temperature(logits.cpu(), targets.cpu())
    model.config.temperature = fitted
    return fitted


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
    search = _policy_search(model, grid)
    fitted_t = _maybe_fit_temperature(
        model, fit_temperature, temperature_loader, val_loader, device
    )
    cache = _EvalCache(model, val_loader, device)

    # start fully conservative -- no exits will fire
    best = [search.seed] * n
    setattr(model.config, search.field, list(best))
    baseline_acc, _ = cache.evaluate()

    # iterate the full grid per exit. an early break would miss thresholds
    # that pass after a transient miss (val accuracy isn't monotone in threshold
    # on small sets). keep the most aggressive passing threshold seen.
    for exit_idx in range(n):
        for thr in search.grid:
            trial = list(best)
            trial[exit_idx] = thr
            setattr(model.config, search.field, list(trial))
            acc, _ = cache.evaluate()
            if baseline_acc - acc <= target_accuracy_drop and search.more_aggressive(
                thr, best[exit_idx]
            ):
                best[exit_idx] = thr

    # clamp any exit that never found a passing threshold from the non-firing
    # search seed back to the in-range disabled value.
    best = [search.no_exit if t == search.seed else t for t in best]
    setattr(model.config, search.field, list(best))
    final_acc, avg_comp = cache.evaluate()
    return CalibrationResult(
        thresholds=best,
        baseline_accuracy=baseline_acc,
        final_accuracy=final_acc,
        avg_computation_used=avg_comp,
        fitted_temperature=fitted_t,
        policy=model.config.routing_policy,
    )


def calibrate_thresholds_for_budget(
    model: EarlyExitWrapper,
    val_loader: DataLoader[Batch],
    target_computation: float = 0.8,
    grid: tuple[float, ...] | None = None,
    device: str = "cpu",
    fit_temperature: bool = False,
    temperature_loader: DataLoader[Batch] | None = None,
) -> CalibrationResult:
    """Find thresholds that meet a compute budget while losing the least accuracy.

    The mirror image of :func:`calibrate_thresholds`: there you state an
    accuracy budget and get the biggest compute saving inside it; here you
    state a compute budget — ``target_computation``, the average fraction of
    the backbone's FLOPs the deployed model may run per sample, in ``(0, 1]``
    — and the search keeps validation accuracy as high as it can while
    meeting it.

    Greedy per exit, earliest first (early exits save the most compute). At
    each exit, among grid values whose average compute is within budget, the
    one with the highest validation accuracy is kept and the search stops;
    later exits stay disabled. If no value at this exit reaches the budget,
    the value with the largest strict compute reduction is kept and the search
    moves to the next exit; an exit that reduces nothing stays disabled. If
    the budget is unattainable after all exits, a ``UserWarning`` is emitted
    and the result carries ``budget_met=False`` with the least-compute
    configuration found (which may be the plain backbone).

    Routing-policy aware exactly like :func:`calibrate_thresholds`, and the
    ``grid`` / ``fit_temperature`` / ``temperature_loader`` parameters behave
    identically.
    """
    if not 0.0 < target_computation <= 1.0:
        raise ValueError(f"target_computation must be in (0, 1], got {target_computation}")
    model = model.to(device)
    n = len(model.config.exit_points)
    search = _policy_search(model, grid)
    fitted_t = _maybe_fit_temperature(
        model, fit_temperature, temperature_loader, val_loader, device
    )
    cache = _EvalCache(model, val_loader, device)

    # start fully conservative -- no exits fire, compute is the full backbone
    best = [search.seed] * n
    setattr(model.config, search.field, list(best))
    baseline_acc, current_comp = cache.evaluate()
    budget_met = current_comp <= target_computation

    for exit_idx in range(n):
        if budget_met:
            break  # budget already met; leave the remaining exits disabled
        # within one exit: the best accuracy among budget-meeting grid values,
        # and the largest strict compute reduction as the fallback when none
        # qualifies
        within: tuple[float, float, float] | None = None  # (acc, thr, comp)
        fallback: tuple[float, float] | None = None  # (comp, thr)
        for thr in search.grid:
            trial = list(best)
            trial[exit_idx] = thr
            setattr(model.config, search.field, list(trial))
            acc, comp = cache.evaluate()
            if comp <= target_computation and (within is None or acc > within[0]):
                within = (acc, thr, comp)
            if comp < current_comp and (fallback is None or comp < fallback[0]):
                fallback = (comp, thr)
        if within is not None:
            best[exit_idx] = within[1]
            current_comp = within[2]
            budget_met = True
        elif fallback is not None:
            best[exit_idx] = fallback[1]
            current_comp = fallback[0]
        # else: no grid value at this exit reduced compute; leave it at the
        # seed so it clamps to the disabled value below

    if not budget_met:
        warnings.warn(
            f"target_computation={target_computation} is unattainable: the "
            f"least-compute configuration found still uses {current_comp:.4f} "
            "of the backbone's FLOPs on the validation set. Returning it with "
            "budget_met=False; consider earlier exit points or a larger budget.",
            UserWarning,
            stacklevel=2,
        )

    # clamp exits still at the non-firing search seed to the in-range
    # disabled value.
    best = [search.no_exit if t == search.seed else t for t in best]
    setattr(model.config, search.field, list(best))
    final_acc, avg_comp = cache.evaluate()
    return CalibrationResult(
        thresholds=best,
        baseline_accuracy=baseline_acc,
        final_accuracy=final_acc,
        avg_computation_used=avg_comp,
        fitted_temperature=fitted_t,
        policy=model.config.routing_policy,
        target_computation=target_computation,
        budget_met=budget_met,
    )
