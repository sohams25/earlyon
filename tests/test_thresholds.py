import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.thresholds import calibrate_thresholds
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone


def _build():
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=[0.8, 0.8],
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


def test_calibration_returns_thresholds_for_each_exit():
    model = _build()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    assert len(result.thresholds) == 2
    assert all(0.0 <= t <= 1.0 for t in result.thresholds)
    assert result.baseline_accuracy >= result.final_accuracy - 0.10  # within target slack
    assert 0.0 < result.avg_computation_used <= 1.0


def test_calibration_writes_thresholds_back_to_config():
    """After calibration the model's config must hold the returned thresholds."""
    model = _build()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    assert model.config.confidence_thresholds == result.thresholds


def test_calibration_thresholds_are_in_grid():
    """Returned thresholds must come from the search grid (or be the initial 1.0)."""
    from earlyon.core.thresholds import DEFAULT_GRID

    model = _build()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    allowed = set(DEFAULT_GRID) | {1.0}
    for t in result.thresholds:
        assert t in allowed


def test_calibration_with_fit_temperature_writes_to_config():
    """When fit_temperature=True, per-head fitted temperatures must land on
    config.temperatures BEFORE the threshold grid search runs."""
    model = _build()
    assert model.config.temperatures == {"e0": 1.0, "e1": 1.0, "final": 1.0}
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    temp_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    result = calibrate_thresholds(
        model,
        val_loader,
        target_accuracy_drop=0.05,
        device="cpu",
        fit_temperature=True,
        temperature_loader=temp_loader,
    )

    # temperature must have been fit and stored (a finite positive scalar; the
    # exact value depends on the data and may legitimately be ~1.0 when there is
    # no calibration signal, so we assert the contract, not a specific value).
    import math

    assert result.temperatures is not None
    assert set(result.temperatures) == {"e0", "e1", "final"}
    for temp in result.temperatures.values():
        assert math.isfinite(temp) and temp > 0.0
    assert model.config.temperatures == result.temperatures
    # deprecated scalar alias still reports the final head's fit
    assert result.fitted_temperature == result.temperatures["final"]
    # per-head fit status is reported
    assert result.temperature_fits is not None
    assert set(result.temperature_fits) == {"e0", "e1", "final"}
    # thresholds still calibrated normally
    assert len(result.thresholds) == 2


def test_calibration_fit_temperature_warns_without_separate_loader():
    """Passing fit_temperature=True without a temperature_loader reuses val_loader,
    which leaks the temperature fit into the threshold accuracy estimate. The
    function must emit a UserWarning so the user is aware."""
    model = _build()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=4)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        calibrate_thresholds(
            model,
            val_loader,
            target_accuracy_drop=0.05,
            device="cpu",
            fit_temperature=True,
        )
    messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any(
        "temperature_loader" in m for m in messages
    ), f"expected leakage warning mentioning temperature_loader; got: {messages}"


def test_calibration_fit_temperature_runs_before_threshold_search():
    """The order matters: fit per-head temperatures, then sweep grids using
    logits/T. We verify by spying on the grid evaluator and asserting every
    evaluate call already saw the fitted temperatures."""
    model = _build()
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 10, (24,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    observed_temps: list[dict[str, float]] = []
    from earlyon.core import thresholds as thresholds_mod

    original_eval = thresholds_mod._EvalCache.evaluate

    def spy_eval(self):
        observed_temps.append(dict(self._model.config.temperatures))
        return original_eval(self)

    thresholds_mod._EvalCache.evaluate = spy_eval
    try:
        result = calibrate_thresholds(
            model,
            val_loader,
            target_accuracy_drop=0.05,
            device="cpu",
            fit_temperature=True,
            temperature_loader=val_loader,
        )
    finally:
        thresholds_mod._EvalCache.evaluate = original_eval

    # every evaluate call (baseline + grid sweep + final) must have seen the
    # already-fitted per-head temperatures, proving the fit ran before the search.
    assert observed_temps, "expected at least one evaluate call"
    assert result.temperatures is not None
    assert all(
        t == result.temperatures for t in observed_temps
    ), f"expected all temps == fitted {result.temperatures}; got {observed_temps}"


def _build_entropy():
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        routing_policy="entropy",
        confidence_thresholds=[0.8, 0.8],
        entropy_thresholds=[0.0, 0.0],  # conservative seed; calibration must move these
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


def test_entropy_calibration_updates_entropy_thresholds_not_confidence():
    """Regression for the silent no-op: calibrating an entropy-routed model must
    write entropy_thresholds (the field the router reads) and leave
    confidence_thresholds untouched. The old confidence-only code wrote the
    wrong field, so this assertion would fail on it."""
    import math

    model = _build_entropy()
    # sharpen e0 so its entropy actually drops below the grid and it can fire;
    # an untrained (near-uniform) head never fires and correctly stays disabled
    with torch.no_grad():
        model.exit_heads["e0"].classifier[-1].weight.mul_(30.0)
        model.exit_heads["e0"].classifier[-1].bias.mul_(30.0)
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    # generous drop so the most aggressive thresholds pass and best[] moves
    result = calibrate_thresholds(model, loader, target_accuracy_drop=1.0, device="cpu")

    assert result.policy == "entropy"
    # the returned thresholds are the entropy field, written back to config
    assert model.config.entropy_thresholds == result.thresholds
    # confidence_thresholds must be untouched by an entropy calibration
    assert model.config.confidence_thresholds == [0.8, 0.8]
    # the sharpened exit must be enabled with a real grid threshold
    assert result.enabled_exits[0] is True
    assert result.thresholds[0] > 0.0
    h_max = math.log(10)
    assert all(0.0 <= t <= h_max + 1e-9 for t in result.thresholds)


def test_entropy_calibration_rejects_out_of_range_custom_grid():
    """A custom entropy grid with values above ln(num_classes) is meaningless and
    must raise rather than silently produce junk thresholds."""
    import math

    import pytest

    model = _build_entropy()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    bad_grid = (0.1, math.log(10) + 5.0)  # second value exceeds H_max
    with pytest.raises(ValueError, match="entropy grid"):
        calibrate_thresholds(model, loader, grid=bad_grid, device="cpu")


def test_entropy_calibration_honors_valid_custom_grid():
    """A valid in-range custom entropy grid must be used as the search space."""
    model = _build_entropy()
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    custom = (0.1, 0.5)  # both < ln(10) ≈ 2.302, in range
    result = calibrate_thresholds(
        model, loader, grid=custom, target_accuracy_drop=1.0, device="cpu"
    )
    allowed = set(custom) | {0.0}  # grid values or the clamped no-exit value
    for t in result.thresholds:
        assert t in allowed
    assert model.config.entropy_thresholds == result.thresholds


def test_confidence_baseline_not_corrupted_by_saturated_head():
    """Regression: a float32-saturated exit head (softmax.max() == 1.0) must NOT
    fire during the baseline measurement. With the old seed=1.0 and the hook's
    `>=`, it would, making baseline_acc reflect the exit head instead of the full
    network. The non-firing seed (2.0) prevents that."""
    model = _build()  # confidence policy, 2 exits
    # force exit e0 to saturate: huge constant logit for class 0 -> softmax max 1.0
    with torch.no_grad():
        model.exit_heads["e0"].classifier[-1].bias.zero_()
        model.exit_heads["e0"].classifier[-1].bias[0] = 1e3

    x = torch.randn(20, 3, 32, 32)
    model.eval()
    # labels = the full network's own predictions, so full-network accuracy is
    # exactly 1.0 while the saturated exit-0 (always class 0) is ~0.1. If the
    # seed fired during baseline, baseline_accuracy would collapse to ~0.1.
    with torch.no_grad():
        y = model.backbone(x).argmax(dim=-1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.0, device="cpu")
    assert result.baseline_accuracy == 1.0, (
        "baseline must equal full-network accuracy (1.0); a saturated head fired "
        f"during the baseline measurement (got {result.baseline_accuracy})"
    )


def test_confidence_calibration_reports_policy():
    """The default confidence path must tag the result with its policy."""
    model = _build()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    assert result.policy == "confidence"


# ---------------- budget calibration ----------------


def _saturate_head(model, name: str) -> None:
    """Make exit `name` always fire: huge constant logit for class 0 gives
    softmax max ~1.0 (confidence policy) and entropy ~0.0 (entropy policy)."""
    with torch.no_grad():
        model.exit_heads[name].classifier[-1].bias.zero_()
        model.exit_heads[name].classifier[-1].bias[0] = 1e3


def _flatten_head(model, name: str) -> None:
    """Make exit `name` never fire under the default grids: zero weights and
    bias give all-zero logits, i.e. a uniform softmax (confidence 0.1 for 10
    classes, entropy = ln 10)."""
    with torch.no_grad():
        model.exit_heads[name].classifier[-1].weight.zero_()
        model.exit_heads[name].classifier[-1].bias.zero_()


def test_budget_calibration_meets_attainable_budget():
    """With a saturated first exit, any threshold fires it, so a generous
    compute budget must be met — and met at the earliest exit, leaving later
    exits disabled."""
    from earlyon.core.thresholds import calibrate_thresholds_for_budget

    model = _build()
    _saturate_head(model, "e0")
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = calibrate_thresholds_for_budget(model, loader, target_computation=0.99, device="cpu")

    assert result.budget_met is True
    assert result.avg_computation_used <= 0.99
    assert len(result.thresholds) == 2
    assert model.config.confidence_thresholds == result.thresholds
    # budget was met at exit 0; exit 1 must stay disabled (no-exit value 1.0)
    assert result.thresholds[1] == 1.0
    assert result.target_computation == 0.99


def test_budget_calibration_unattainable_budget_warns_and_flags():
    """When no threshold combination can reach the budget (uniform heads never
    fire), the function must warn, set budget_met=False, and leave the
    never-helping exits disabled rather than aggressive."""
    from earlyon.core.thresholds import calibrate_thresholds_for_budget

    model = _build()
    _flatten_head(model, "e0")
    _flatten_head(model, "e1")
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = calibrate_thresholds_for_budget(
            model, loader, target_computation=0.5, device="cpu"
        )

    assert result.budget_met is False
    assert result.avg_computation_used == 1.0
    # a threshold that never fires reduces nothing; both exits stay disabled
    assert result.thresholds == [1.0, 1.0]
    messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("target_computation" in m for m in messages), messages


def test_budget_calibration_trivial_budget_keeps_exits_disabled():
    """A budget of 1.0 is met by the plain backbone; no exit should be made
    aggressive just because the machinery ran."""
    from earlyon.core.thresholds import calibrate_thresholds_for_budget

    model = _build()
    _saturate_head(model, "e0")  # even a head that WOULD fire must stay disabled
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = calibrate_thresholds_for_budget(model, loader, target_computation=1.0, device="cpu")

    assert result.budget_met is True
    assert result.thresholds == [1.0, 1.0]
    assert result.enabled_exits == [False, False]
    # explicit enablement: a disabled exit can no longer fire even with a
    # float32-saturated softmax (confidence exactly 1.0), so the average
    # compute is exactly the full backbone.
    assert result.avg_computation_used == 1.0


def test_budget_calibration_validates_target():
    import pytest

    from earlyon.core.thresholds import calibrate_thresholds_for_budget

    model = _build()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    for bad in (0.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="target_computation"):
            calibrate_thresholds_for_budget(model, loader, target_computation=bad, device="cpu")


def test_budget_calibration_entropy_policy_writes_entropy_field():
    """Budget calibration must be policy-aware exactly like the accuracy
    version: an entropy-routed model gets entropy_thresholds written and
    confidence_thresholds left alone."""
    from earlyon.core.thresholds import calibrate_thresholds_for_budget

    model = _build_entropy()
    _saturate_head(model, "e0")  # saturated head => entropy ~0, fires at any grid value
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = calibrate_thresholds_for_budget(model, loader, target_computation=0.99, device="cpu")

    assert result.policy == "entropy"
    assert result.budget_met is True
    assert model.config.entropy_thresholds == result.thresholds
    assert model.config.confidence_thresholds == [0.8, 0.8]
    # exit 0 calibrated to a real grid value; exit 1 disabled at the entropy
    # no-exit value 0.0
    assert result.thresholds[0] > 0.0
    assert result.thresholds[1] == 0.0


def test_calibration_without_fit_temperature_leaves_config_unchanged():
    """The default path (fit_temperature=False) must not touch config.temperatures."""
    model = _build()
    model.config.temperatures = {"e0": 1.5, "e1": 1.5, "final": 1.5}  # user-provided
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=4)

    calibrate_thresholds(
        model,
        val_loader,
        target_accuracy_drop=0.05,
        device="cpu",
    )

    assert model.config.temperatures == {"e0": 1.5, "e1": 1.5, "final": 1.5}


def test_calibration_rejects_empty_grid():
    """An empty custom grid means nothing is searched: calibration 'succeeds'
    with all exits disabled and the user ships a model that never exits. Both
    calibrators must refuse it."""
    import pytest

    from earlyon.core.thresholds import calibrate_thresholds_for_budget

    model = _build()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    with pytest.raises(ValueError, match="grid"):
        calibrate_thresholds(model, loader, grid=(), device="cpu")
    with pytest.raises(ValueError, match="grid"):
        calibrate_thresholds_for_budget(model, loader, grid=(), device="cpu")


def test_eval_cache_matches_real_router_exactly():
    """The vectorized calibration evaluator must agree with the real
    per-sample routing path on accuracy AND avg compute, for both policies,
    across threshold settings, temperatures, and adversarial heads. This is
    the guard that lets calibration run one batched pass instead of one full
    validation pass per grid point."""
    from earlyon.core.thresholds import _EvalCache, _evaluate_with_router

    torch.manual_seed(7)
    for build, policy in ((_build, "confidence"), (_build_entropy, "entropy")):
        model = build()
        # one saturated head + one normal head exercises the knife edges
        with torch.no_grad():
            model.exit_heads["e0"].classifier[-1].bias.zero_()
            model.exit_heads["e0"].classifier[-1].bias[0] = 1e3
        x = torch.randn(40, 3, 32, 32)
        y = torch.randint(0, 10, (40,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        cache = _EvalCache(model, loader, "cpu")

        settings = [
            # (conf_thr, ent_thr, temps, enabled)
            ([1.0, 1.0], [0.0, 0.0], (1.0, 1.0, 1.0), [False, False]),  # all disabled
            ([1.0, 1.0], [0.0, 0.0], (1.0, 1.0, 1.0), [True, True]),  # knife-edge thresholds
            ([0.8, 0.5], [0.5, 1.2], (1.0, 1.0, 1.0), [True, True]),  # mid-grid
            ([0.5, 0.5], [2.0, 2.0], (2.5, 0.7, 1.3), [True, True]),  # per-head temps
            ([0.5, 0.5], [2.0, 2.0], (2.5, 0.7, 1.3), [False, True]),  # first exit off
        ]
        for conf_thr, ent_thr, temps, enabled in settings:
            model.config.confidence_thresholds = list(conf_thr)
            model.config.entropy_thresholds = list(ent_thr)
            model.config.temperatures = {"e0": temps[0], "e1": temps[1], "final": temps[2]}
            model.config.enabled_exits = list(enabled)
            real = _evaluate_with_router(model, loader, "cpu")
            sim = cache.evaluate()
            # accuracy must match exactly (integer counts); avg compute may
            # differ by float summation order, so allow one ulp of slack
            assert sim[0] == real[0] and abs(sim[1] - real[1]) < 1e-12, (
                f"policy={policy} conf={conf_thr} ent={ent_thr} T={temps} "
                f"enabled={enabled}: sim={sim} real={real}"
            )


# ---------------- staged pipeline (v0.3) ----------------


def test_calibration_rejects_empty_loader():
    """An empty calibration loader must fail fast, not silently produce a
    policy chosen from zero samples (or hang)."""
    import pytest

    model = _build()
    empty = DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="no batches"):
        calibrate_thresholds(model, empty, device="cpu")


def test_calibration_result_carries_rich_metadata():
    model = _build()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    assert result.num_samples == 16
    assert result.objective == "accuracy_budget"
    assert result.method == "greedy-coordinate-grid"
    assert result.schema_version == 2
    assert len(result.enabled_exits) == 2
    assert abs(sum(result.exit_distribution.values()) - 1.0) < 1e-9
    assert abs(result.accuracy_delta - (result.baseline_accuracy - result.final_accuracy)) < 1e-12
    assert result.target_accuracy_drop == 0.05
    # enablement written back to the model config
    assert model.config.enabled_exits == result.enabled_exits


def test_calibration_disables_exit_with_no_passing_threshold():
    """A head that is pure noise (uniform logits never reach the grid) must end
    up explicitly disabled, not parked on a sentinel threshold."""
    model = _build()
    _flatten_head(model, "e0")
    _flatten_head(model, "e1")
    x = torch.randn(16, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model.backbone(x).argmax(dim=-1)  # full-network labels: baseline acc 1.0
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.0, device="cpu")
    assert result.enabled_exits == [False, False]
    assert model.config.enabled_exits == [False, False]
    assert result.exit_distribution == {"final": 1.0}


def test_fit_head_temperatures_can_differ_per_head():
    """Heads with different miscalibration must receive different fitted
    temperatures — the P0 defect was one temperature fit from the final
    classifier reused everywhere."""
    from earlyon.core.thresholds import collect_head_logits, fit_head_temperatures

    torch.manual_seed(3)
    model = _build()
    # make e0 wildly overconfident (large weight scale); leave e1/final alone
    with torch.no_grad():
        model.exit_heads["e0"].classifier[-1].weight.mul_(30.0)
        model.exit_heads["e0"].classifier[-1].bias.mul_(30.0)
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    cache = collect_head_logits(model, loader, "cpu")
    fits = fit_head_temperatures(cache)
    assert set(fits) == {"e0", "e1", "final"}
    t_e0 = fits["e0"].temperature
    t_final = fits["final"].temperature
    # the overconfident head needs a much larger temperature than the final head
    assert t_e0 > t_final * 2, (t_e0, t_final)


def test_collect_head_logits_shapes_and_order():
    from earlyon.core.thresholds import collect_head_logits

    model = _build()
    x = torch.randn(12, 3, 32, 32)
    y = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    cache = collect_head_logits(model, loader, "cpu")
    assert cache.head_names == ["e0", "e1", "final"]
    assert [t.shape for t in cache.logits] == [torch.Size([12, 10])] * 3
    assert cache.num_samples == 12
    assert cache.flops[-1].item() == 1.0
    # cumulative fractions are monotone non-decreasing
    flops = cache.flops.tolist()
    assert all(a <= b for a, b in zip(flops, flops[1:]))


def test_collect_head_logits_restores_training_mode():
    from earlyon.core.thresholds import collect_head_logits

    model = _build()
    model.train()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    collect_head_logits(model, loader, "cpu")
    assert model.training is True
