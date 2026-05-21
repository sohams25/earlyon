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
