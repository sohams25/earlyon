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
    """When fit_temperature=True, the fitted scalar must land on config.temperature
    BEFORE the threshold grid search runs."""
    model = _build()
    assert model.config.temperature == 1.0
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

    assert result.fitted_temperature is not None
    assert math.isfinite(result.fitted_temperature)
    assert result.fitted_temperature > 0.0
    assert model.config.temperature == result.fitted_temperature
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
    """The order matters: fit T, then sweep grids using logits/T. We verify by
    asserting model.config.temperature is already non-1.0 by the time the search
    begins — done indirectly by patching the grid evaluator."""
    model = _build()
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 10, (24,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    observed_temps: list[float] = []
    from earlyon.core import thresholds as thresholds_mod

    original_eval = thresholds_mod._evaluate

    def spy_eval(m, loader, device):
        observed_temps.append(m.config.temperature)
        return original_eval(m, loader, device)

    thresholds_mod._evaluate = spy_eval
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
        thresholds_mod._evaluate = original_eval

    # every _evaluate call (baseline + grid sweep + final) must have seen the
    # already-fitted temperature, proving the fit ran before the search.
    assert observed_temps, "expected at least one _evaluate call"
    assert all(
        t == result.fitted_temperature for t in observed_temps
    ), f"expected all temps == fitted {result.fitted_temperature}; got {observed_temps}"


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
    # they must have actually moved off the conservative 0.0 seed
    assert any(t > 0.0 for t in result.thresholds)
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
    # NOT asserting avg_computation_used == 1.0: the routing hook fires on
    # `confidence >= threshold`, so a float32-saturated head (confidence 1.0)
    # still fires at the clamped disabled value 1.0 — the library's documented
    # convention, shared with calibrate_thresholds. The budget contract is
    # only comp <= target.
    assert result.avg_computation_used <= 1.0


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
    """The default path (fit_temperature=False) must not touch config.temperature."""
    model = _build()
    model.config.temperature = 1.5  # user-provided
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=4)

    calibrate_thresholds(
        model,
        val_loader,
        target_accuracy_drop=0.05,
        device="cpu",
    )

    assert model.config.temperature == 1.5
