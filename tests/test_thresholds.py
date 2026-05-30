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

    # temperature must have been fit and stored
    assert model.config.temperature != 1.0
    assert model.config.temperature > 0.0
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
    assert any("temperature_loader" in m for m in messages), (
        f"expected leakage warning mentioning temperature_loader; got: {messages}"
    )


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
        calibrate_thresholds(
            model,
            val_loader,
            target_accuracy_drop=0.05,
            device="cpu",
            fit_temperature=True,
            temperature_loader=val_loader,
        )
    finally:
        thresholds_mod._evaluate = original_eval

    # every _evaluate call must have seen the fitted (non-1.0) temperature
    assert observed_temps, "expected at least one _evaluate call"
    assert all(t != 1.0 for t in observed_temps), (
        f"expected all temps != 1.0 (search ran after fit); got {observed_temps}"
    )


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
