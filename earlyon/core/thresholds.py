"""Threshold calibration: staged, testable, greedy-by-construction.

The pipeline is four separate stages, each independently callable and tested:

1. **Collect** — :func:`collect_head_logits` runs the calibration loader once
   through the all-exits (training-mode) forward and caches every head's raw
   logits plus targets. The network runs exactly once.
2. **Fit temperatures** — :func:`fit_head_temperatures` fits one temperature
   per head (every exit head *and* the final classifier) on cached logits via
   post-hoc temperature scaling (Guo et al. 2017). Each head is a different
   classifier with its own miscalibration; a single shared temperature is not
   defensible.
3. **Search thresholds** — a greedy coordinate-descent sweep over a fixed
   grid, simulated vectorially against the cache. It is *not* a joint or
   globally optimal search and is not claimed to be one; see
   ``docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md``.
4. **Evaluate** — the selected policy is scored on the calibration split
   (accuracy, estimated compute, exit distribution). Held-out *test*
   evaluation is a separate concern: use
   :func:`earlyon.benchmarking.evaluate` on a test loader; the threshold
   search never sees test labels.

Two public calibrators share the search:

* :func:`calibrate_thresholds` — accuracy budget. For each exit (low-index to
  high-index), lower the threshold through the grid; keep the most aggressive
  value whose accuracy drop stays within ``target_accuracy_drop``.
* :func:`calibrate_thresholds_for_budget` — compute budget. For each exit,
  keep the threshold that brings average estimated compute within
  ``target_computation`` while losing the least accuracy.

Exits for which no grid value passes are left **disabled** via
``config.enabled_exits`` — an explicit boolean, not a threshold sentinel, so
a saturated softmax (confidence exactly 1.0) can never fire a disabled exit.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import DataLoader

from earlyon.core.temperature import TemperatureFitResult, fit_temperature_full
from earlyon.core.types import FINAL_HEAD, Batch, exit_label
from earlyon.core.wrappers import EarlyExitWrapper, _safe_temperature

CALIBRATION_SCHEMA_VERSION = 2

# confidence grid: ordered conservative -> aggressive (lower threshold fires more)
DEFAULT_GRID: tuple[float, ...] = (0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)

# entropy grid: fractions of H_max = ln(num_classes), ordered conservative ->
# aggressive (higher entropy threshold fires more, since exit fires on H <= thr)
DEFAULT_ENTROPY_FRACTIONS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass
class CalibrationResult:
    """Everything a calibration run decided and measured.

    ``avg_computation_used`` is the average *estimated backbone FLOPs
    fraction* over the calibration set (see
    ``InferenceResult.estimated_backbone_flops_fraction`` for the estimator's
    limitations); it is not a wall-clock or measured-FLOPs quantity.

    ``baseline_accuracy`` is the routed accuracy with every exit disabled,
    i.e. the final classifier's accuracy on the calibration split.
    ``final_accuracy`` is the routed accuracy under the selected policy on the
    same split. Both are calibration-split numbers — report test-set accuracy
    from a separate held-out loader.
    """

    thresholds: list[float]
    baseline_accuracy: float
    final_accuracy: float
    avg_computation_used: float
    enabled_exits: list[bool] = field(default_factory=list)
    accuracy_delta: float = 0.0
    exit_distribution: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    fitted_temperature: float | None = None  # deprecated: final head's fitted T
    temperatures: dict[str, float] | None = None  # per-head fitted temperatures
    temperature_fits: dict[str, TemperatureFitResult] | None = None
    policy: str = "confidence"  # which routing policy the thresholds calibrate
    objective: str = "accuracy_budget"  # "accuracy_budget" | "compute_budget"
    method: str = "greedy-coordinate-grid"  # honestly named: not a joint optimum
    target_accuracy_drop: float | None = None
    target_computation: float | None = None  # set by budget calibration
    budget_met: bool = True  # False when target_computation was unattainable
    schema_version: int = CALIBRATION_SCHEMA_VERSION


# ---------------- stage 1: collect ----------------


@dataclass(frozen=True)
class HeadLogits:
    """Cached raw (temperature-1) logits for every head on one dataset pass.

    ``head_names`` is the exit names in forward order plus ``"final"`` last;
    ``logits[i]`` is the ``(N, num_classes)`` CPU tensor for ``head_names[i]``;
    ``flops`` is each head's cumulative estimated backbone-FLOPs fraction
    (1.0 for the final head).
    """

    head_names: list[str]
    logits: list[torch.Tensor]
    targets: torch.Tensor
    flops: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.targets.shape[0])


def collect_head_logits(
    model: EarlyExitWrapper, loader: DataLoader[Batch], device: str
) -> HeadLogits:
    """One batched training-mode pass caching every head's logits + targets.

    Training mode never fires an exit (the hook only appends logits), so
    every head — including disabled ones — is captured, and temperatures are
    irrelevant here (they apply at softmax time, not logit time).

    Raises ``ValueError`` on an empty loader: silently calibrating on zero
    samples would produce a policy chosen from nothing.
    """
    model = model.to(device)
    per_head: list[list[torch.Tensor]] = []
    targets: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for images, tgt in loader:
                outputs = model(images.to(device), mode="training")
                if not per_head:
                    per_head = [[] for _ in outputs]
                for idx, logits in enumerate(outputs):
                    per_head[idx].append(logits.detach().cpu())
                targets.append(tgt.cpu())
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError(
            "calibration loader yielded no batches; calibration needs a "
            "non-empty validation/calibration split"
        )
    exit_flops = [model._flops_at[ep.layer_name] for ep in model.config.exit_points]
    return HeadLogits(
        head_names=model.config.head_names,
        logits=[torch.cat(chunks) for chunks in per_head],
        targets=torch.cat(targets),
        flops=torch.tensor(exit_flops + [1.0], dtype=torch.float64),
    )


# ---------------- stage 2: fit per-head temperatures ----------------


def fit_head_temperatures(cache: HeadLogits) -> dict[str, TemperatureFitResult]:
    """Fit one temperature per head (exits and final) from cached logits.

    Fitting is per-head NLL minimisation; each result carries its
    convergence/fallback status. Callers decide what to do with fallbacks —
    :func:`calibrate_thresholds` warns and keeps the safe 1.0.
    """
    return {
        name: fit_temperature_full(logits, cache.targets)
        for name, logits in zip(cache.head_names, cache.logits)
    }


# ---------------- stage 3/4: simulate + evaluate ----------------


@dataclass(frozen=True)
class PolicyEvaluation:
    """Routed metrics for one (thresholds, enabled, temperatures) setting."""

    accuracy: float
    avg_flops_fraction: float
    exit_distribution: dict[str, float]


def _evaluate_with_router(
    model: EarlyExitWrapper, loader: DataLoader[Batch], device: str
) -> tuple[float, float]:
    """Ground truth: route every sample through the real inference path.

    Kept as the oracle that pins :func:`simulate_policy` correctness in the
    test suite; calibration itself uses the cache."""
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
                comp += result.estimated_backbone_flops_fraction
                total += 1
    return correct / max(total, 1), comp / max(total, 1)


def simulate_policy(cache: HeadLogits, model: EarlyExitWrapper) -> PolicyEvaluation:
    """Vectorially replay the router's decision rule against cached logits.

    Reproduces the routing math exactly: per-head temperature, softmax in
    float32 (as the hook computes it), criterion comparison in float64 (as the
    hook's ``.item()`` produces), first firing *enabled* exit claims the
    sample. A test pins equivalence against :func:`_evaluate_with_router`.
    """
    cfg = model.config
    n_exits = len(cfg.exit_points)
    n = cache.num_samples
    chosen = torch.full((n,), n_exits, dtype=torch.long)
    for idx in range(n_exits - 1, -1, -1):  # earlier exits override later ones
        if not cfg.enabled_exits[idx]:
            continue
        temp = _safe_temperature(cfg.temperatures[cfg.exit_points[idx].name])
        probs = torch.softmax(cache.logits[idx] / temp, dim=-1)
        if cfg.routing_policy == "confidence":
            confidence = probs.max(dim=-1).values.double()
            fires = confidence >= cfg.confidence_thresholds[idx]
        else:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            fires = entropy.double() <= cfg.entropy_thresholds[idx]
        chosen[fires] = idx
    preds = torch.stack([h.argmax(dim=-1) for h in cache.logits])
    pred = preds[chosen, torch.arange(n)]
    acc = (pred == cache.targets).double().mean().item()
    comp = cache.flops[chosen].mean().item()
    counts = torch.bincount(chosen, minlength=n_exits + 1)
    dist = {
        exit_label(i if i < n_exits else -1): counts[i].item() / n
        for i in range(n_exits + 1)
        if counts[i] > 0
    }
    return PolicyEvaluation(accuracy=acc, avg_flops_fraction=comp, exit_distribution=dist)


class _EvalCache:
    """Adapter binding a :class:`HeadLogits` cache to a model for the greedy
    search loops. ``evaluate()`` returns ``(accuracy, avg flops fraction)``
    under the model's *current* config."""

    def __init__(self, model: EarlyExitWrapper, loader: DataLoader[Batch], device: str) -> None:
        self._model = model
        self.cache = collect_head_logits(model, loader, device)

    def evaluate(self) -> tuple[float, float]:
        ev = simulate_policy(self.cache, self._model)
        return ev.accuracy, ev.avg_flops_fraction

    def evaluate_full(self) -> PolicyEvaluation:
        return simulate_policy(self.cache, self._model)


# ---------------- search setup ----------------


@dataclass(frozen=True)
class _PolicySearch:
    """Per-policy search setup. ``disabled_placeholder`` is the value recorded
    in the threshold list for exits that end up disabled — routing never reads
    it (``enabled_exits`` is authoritative), it only keeps the list fully
    populated and in-range. ``more_aggressive`` decides which passing
    threshold to keep (smaller for confidence, larger for entropy)."""

    field: str
    disabled_placeholder: float
    grid: tuple[float, ...]
    more_aggressive: Callable[[float, float], bool]


def _policy_search(model: EarlyExitWrapper, grid: tuple[float, ...] | None) -> _PolicySearch:
    if grid is not None and len(grid) == 0:
        # an empty grid searches nothing: calibration would "succeed" with all
        # exits disabled and the model would never exit early
        raise ValueError("custom grid must be non-empty (or None for the default)")
    policy = model.config.routing_policy
    if policy == "confidence":
        if grid is not None and any(not math.isfinite(t) or not 0.0 <= t <= 1.0 for t in grid):
            raise ValueError(f"confidence grid values must lie in [0, 1]; got {grid}")
        return _PolicySearch(
            field="confidence_thresholds",
            disabled_placeholder=1.0,
            grid=DEFAULT_GRID if grid is None else grid,
            more_aggressive=lambda new, cur: new < cur,
        )
    if policy == "entropy":
        h_max = math.log(max(model.config.num_classes, 2))
        if grid is None:
            search_grid = tuple(round(f * h_max, 6) for f in DEFAULT_ENTROPY_FRACTIONS)
        else:
            if any(not math.isfinite(t) or t < 0 or t > h_max + 1e-9 for t in grid):
                raise ValueError(
                    f"entropy grid values must lie in [0, ln(num_classes)] = "
                    f"[0, {h_max:.4f}]; got {grid}"
                )
            search_grid = grid
        return _PolicySearch(
            field="entropy_thresholds",
            disabled_placeholder=0.0,
            grid=search_grid,
            more_aggressive=lambda new, cur: new > cur,
        )
    raise ValueError(  # pragma: no cover - EarlyExitConfig already validates the policy
        f"unsupported routing_policy {policy!r}"
    )


def _maybe_fit_temperatures(
    model: EarlyExitWrapper,
    fit_temperature: bool,
    temperature_loader: DataLoader[Batch] | None,
    val_loader: DataLoader[Batch],
    device: str,
) -> dict[str, TemperatureFitResult] | None:
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

    cache = collect_head_logits(model, temperature_loader, device)
    fits = fit_head_temperatures(cache)
    bad = [name for name, fit in fits.items() if fit.fallback]
    if bad:
        warnings.warn(
            f"temperature fitting diverged for head(s) {bad}; those heads keep "
            "the last finite iterate (1.0 if none) and are effectively "
            "uncalibrated. See TemperatureFitResult.fallback in the result.",
            UserWarning,
            stacklevel=3,
        )
    model.config.temperatures = {name: fit.temperature for name, fit in fits.items()}
    model.config.validate()
    return fits


# ---------------- public calibrators ----------------


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

    Exits for which no grid value stays inside the accuracy budget are left
    disabled in ``model.config.enabled_exits``. Calibration decides enablement
    from scratch: any pre-existing enabled/disabled state is overwritten,
    because the search's job is to select the deployment policy.

    ``val_loader`` is the *calibration* split. Do not pass your test set: the
    threshold search consumes its labels. Evaluate the calibrated model on a
    separate test loader with :func:`earlyon.benchmarking.evaluate`.

    Parameters
    ----------
    target_accuracy_drop:
        Maximum acceptable drop (on the calibration split) from the
        final-classifier baseline, in [0, 1].
    grid:
        Optional custom search grid. If ``None`` a policy-appropriate default
        is used. Confidence grids must lie in [0, 1]; entropy grids in
        ``[0, ln(num_classes)]``.
    fit_temperature:
        If True, fit post-hoc per-head temperature scaling (every exit head
        and the final classifier) before the threshold search. Fitted values
        land on ``model.config.temperatures``.
    temperature_loader:
        DataLoader used to fit the temperatures. If ``None`` and
        ``fit_temperature`` is True, ``val_loader`` is reused — this leaks the
        temperature fit into the threshold accuracy estimate, so a warning
        is emitted. Prefer a separate held-out split when available.
    """
    if not math.isfinite(target_accuracy_drop) or not 0.0 <= target_accuracy_drop <= 1.0:
        raise ValueError(f"target_accuracy_drop must be in [0, 1], got {target_accuracy_drop}")
    model = model.to(device)
    cfg = model.config
    n = len(cfg.exit_points)
    search = _policy_search(model, grid)
    fits = _maybe_fit_temperatures(model, fit_temperature, temperature_loader, val_loader, device)
    cache = _EvalCache(model, val_loader, device)

    # baseline: every exit disabled — the final classifier's routed accuracy
    best = [search.disabled_placeholder] * n
    enabled = [False] * n
    cfg.enabled_exits = list(enabled)
    setattr(cfg, search.field, list(best))
    baseline_acc, _ = cache.evaluate()

    # iterate the full grid per exit. an early break would miss thresholds
    # that pass after a transient miss (val accuracy isn't monotone in threshold
    # on small sets). keep the most aggressive passing threshold seen. A
    # threshold that fires no calibration sample (compute unchanged vs. the
    # exit disabled) is useless — the exit stays disabled instead of shipping
    # an enabled-but-idle exit that might fire on out-of-distribution inputs.
    for exit_idx in range(n):
        setattr(cfg, search.field, list(best))
        cfg.enabled_exits = list(enabled)
        _, comp_without = cache.evaluate()
        chosen: float | None = None
        for thr in search.grid:
            trial = list(best)
            trial[exit_idx] = thr
            trial_enabled = list(enabled)
            trial_enabled[exit_idx] = True
            setattr(cfg, search.field, list(trial))
            cfg.enabled_exits = trial_enabled
            acc, comp = cache.evaluate()
            useful = comp < comp_without
            if (
                useful
                and baseline_acc - acc <= target_accuracy_drop
                and (chosen is None or search.more_aggressive(thr, chosen))
            ):
                chosen = thr
        if chosen is not None:
            best[exit_idx] = chosen
            enabled[exit_idx] = True

    setattr(cfg, search.field, list(best))
    cfg.enabled_exits = list(enabled)
    cfg.validate()
    final = cache.evaluate_full()
    return CalibrationResult(
        thresholds=best,
        enabled_exits=list(enabled),
        baseline_accuracy=baseline_acc,
        final_accuracy=final.accuracy,
        avg_computation_used=final.avg_flops_fraction,
        accuracy_delta=baseline_acc - final.accuracy,
        exit_distribution=final.exit_distribution,
        num_samples=cache.cache.num_samples,
        fitted_temperature=fits[FINAL_HEAD].temperature if fits else None,
        temperatures=dict(cfg.temperatures) if fits else None,
        temperature_fits=fits,
        policy=cfg.routing_policy,
        objective="accuracy_budget",
        target_accuracy_drop=target_accuracy_drop,
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
    state a compute budget — ``target_computation``, the average *estimated*
    fraction of the backbone's FLOPs the deployed model may run per sample, in
    ``(0, 1]`` — and the search keeps validation accuracy as high as it can
    while meeting it.

    Greedy per exit, earliest first (early exits save the most compute). At
    each exit, among grid values whose average estimated compute is within
    budget, the one with the highest validation accuracy is kept and the
    search stops; later exits stay disabled. If no value at this exit reaches
    the budget, the value with the largest strict compute reduction is kept
    and the search moves to the next exit; an exit that reduces nothing stays
    disabled. If the budget is unattainable after all exits, a ``UserWarning``
    is emitted and the result carries ``budget_met=False`` with the
    least-compute configuration found (which may be the plain backbone).

    Routing-policy aware exactly like :func:`calibrate_thresholds`, and the
    ``grid`` / ``fit_temperature`` / ``temperature_loader`` parameters behave
    identically.
    """
    if not math.isfinite(target_computation) or not 0.0 < target_computation <= 1.0:
        raise ValueError(f"target_computation must be in (0, 1], got {target_computation}")
    model = model.to(device)
    cfg = model.config
    n = len(cfg.exit_points)
    search = _policy_search(model, grid)
    fits = _maybe_fit_temperatures(model, fit_temperature, temperature_loader, val_loader, device)
    cache = _EvalCache(model, val_loader, device)

    # start fully conservative -- no exits fire, compute is the full backbone
    best = [search.disabled_placeholder] * n
    enabled = [False] * n
    cfg.enabled_exits = list(enabled)
    setattr(cfg, search.field, list(best))
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
            trial_enabled = list(enabled)
            trial_enabled[exit_idx] = True
            setattr(cfg, search.field, list(trial))
            cfg.enabled_exits = trial_enabled
            acc, comp = cache.evaluate()
            if comp <= target_computation and (within is None or acc > within[0]):
                within = (acc, thr, comp)
            if comp < current_comp and (fallback is None or comp < fallback[0]):
                fallback = (comp, thr)
        if within is not None:
            best[exit_idx] = within[1]
            enabled[exit_idx] = True
            current_comp = within[2]
            budget_met = True
        elif fallback is not None:
            best[exit_idx] = fallback[1]
            enabled[exit_idx] = True
            current_comp = fallback[0]
        # else: no grid value at this exit reduced compute; leave it disabled

    if not budget_met:
        warnings.warn(
            f"target_computation={target_computation} is unattainable: the "
            f"least-compute configuration found still uses {current_comp:.4f} "
            "of the backbone's estimated FLOPs on the validation set. Returning "
            "it with budget_met=False; consider earlier exit points or a larger "
            "budget.",
            UserWarning,
            stacklevel=2,
        )

    setattr(cfg, search.field, list(best))
    cfg.enabled_exits = list(enabled)
    cfg.validate()
    final = cache.evaluate_full()
    return CalibrationResult(
        thresholds=best,
        enabled_exits=list(enabled),
        baseline_accuracy=baseline_acc,
        final_accuracy=final.accuracy,
        avg_computation_used=final.avg_flops_fraction,
        accuracy_delta=baseline_acc - final.accuracy,
        exit_distribution=final.exit_distribution,
        num_samples=cache.cache.num_samples,
        fitted_temperature=fits[FINAL_HEAD].temperature if fits else None,
        temperatures=dict(cfg.temperatures) if fits else None,
        temperature_fits=fits,
        policy=cfg.routing_policy,
        objective="compute_budget",
        target_computation=target_computation,
        budget_met=budget_met,
    )
