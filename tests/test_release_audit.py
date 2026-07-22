"""Independent release-readiness audit tests.

Adversarial checks of the hardening claims, covering gaps the original
suites left: batched-path per-head temperatures, a decisive
global-vs-per-head temperature counterfactual, NaN-emitting heads during
calibration, calibration determinism, and proof that the staged runtime
genuinely skips later computation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.thresholds import (
    calibrate_thresholds,
    collect_head_logits,
    fit_head_temperatures,
)
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models import custom_ee
from earlyon.staged import staged_model
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone


def _build(thresholds=(0.95, 0.95)):
    torch.manual_seed(0)
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
        confidence_thresholds=list(thresholds),
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


# ---------------- A. per-head temperatures ----------------


def test_single_global_temperature_would_misroute_where_per_head_does_not():
    """The decisive counterfactual for per-head temperature scaling.

    Exit e0 is an overconfident junk head (huge constant logits for class 0);
    the labels are the full network's own predictions, so routing to the
    final classifier is always correct and routing through e0 is ~10%.

    A per-head fit assigns e0 a huge temperature, deflating its confidence
    below the threshold: routed accuracy stays perfect. Reusing the final
    head's (near-1.0) temperature for e0 — the pre-hardening behavior —
    leaves e0 saturated, so it fires and accuracy collapses. One global
    temperature cannot be right for both heads.
    """
    torch.manual_seed(0)
    model = _build(thresholds=(0.95, 0.95))
    with torch.no_grad():
        model.exit_heads["e0"].classifier[-1].weight.zero_()
        model.exit_heads["e0"].classifier[-1].bias.zero_()
        model.exit_heads["e0"].classifier[-1].bias[0] = 50.0  # saturated junk
        # e1 flat so it never interferes
        model.exit_heads["e1"].classifier[-1].weight.zero_()
        model.exit_heads["e1"].classifier[-1].bias.zero_()
    model.eval()
    x = torch.randn(40, 3, 32, 32)
    with torch.no_grad():
        y = model.backbone(x).argmax(dim=-1)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    cache = collect_head_logits(model, loader, "cpu")
    fits = fit_head_temperatures(cache)
    t_e0 = fits["e0"].temperature
    t_final = fits["final"].temperature
    assert t_e0 > 10 * t_final, (t_e0, t_final)  # radically different calibration needs

    def routed_accuracy() -> float:
        correct = 0
        for i in range(x.shape[0]):
            result = model(x[i : i + 1], mode="inference")
            correct += int(result.prediction.argmax().item() == int(y[i]))
        return correct / x.shape[0]

    # per-head temperatures: e0 deflated below 0.95, never fires -> perfect
    model.config.temperatures = {k: v.temperature for k, v in fits.items()}
    assert routed_accuracy() == 1.0

    # counterfactual: one global temperature (the final head's fit) for all
    # heads — e0 stays saturated, fires its junk prediction, accuracy tanks
    model.config.temperatures = {"e0": t_final, "e1": t_final, "final": t_final}
    assert routed_accuracy() < 0.5


def test_batched_routing_uses_per_head_temperatures():
    """The batched path shares the hook, but nothing pinned it: flattening
    only e0 via its own temperature must stop the batch from exiting at e0."""
    model = _build(thresholds=(0.9, 0.9))
    with torch.no_grad():
        model.exit_heads["e0"].classifier[-1].weight.zero_()
        model.exit_heads["e0"].classifier[-1].bias.zero_()
        model.exit_heads["e0"].classifier[-1].bias[0] = 20.0
        model.exit_heads["e1"].classifier[-1].weight.zero_()
        model.exit_heads["e1"].classifier[-1].bias.zero_()
    model.eval()
    x = torch.randn(4, 3, 32, 32)

    model.config.temperatures = {"e0": 1.0, "e1": 1.0, "final": 1.0}
    assert model.forward_inference_batched(x).exit_taken == 0

    model.config.temperatures = {"e0": 100.0, "e1": 1.0, "final": 1.0}
    assert model.forward_inference_batched(x).exit_taken == -1


# ---------------- C. calibration edge cases + determinism ----------------


def test_calibration_is_deterministic():
    torch.manual_seed(5)
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 10, (24,))

    results = []
    for _ in range(2):
        torch.manual_seed(7)
        model = _build()
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        r = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
        results.append((r.thresholds, r.enabled_exits, r.final_accuracy, r.exit_distribution))
    assert results[0] == results[1]


def test_calibration_survives_nan_emitting_head():
    """A head whose logits are NaN (e.g. numerically broken fine-tune) must
    not crash the threshold search or get enabled: NaN comparisons are False,
    so it never fires and stays disabled."""
    model = _build()
    with torch.no_grad():
        model.exit_heads["e0"].classifier[-1].weight.fill_(float("nan"))
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    assert result.enabled_exits[0] is False


def test_temperature_fit_rejects_nan_logits_cleanly():
    from earlyon.core.temperature import fit_temperature_full

    logits = torch.tensor([[float("nan"), 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_temperature_full(logits, torch.tensor([0, 1]))


def test_calibration_single_class_loader_works():
    """All-one-class calibration data is degenerate but legal; it must not
    crash and the routed model must still work."""
    model = _build()
    x = torch.randn(12, 3, 32, 32)
    y = torch.zeros(12, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    result = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    assert len(result.thresholds) == 2
    assert 0.0 <= result.final_accuracy <= 1.0


# ---------------- I. staged runtime genuinely skips computation ----------------


class _CountingModule(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(x)


def test_staged_runtime_skips_later_stages_when_exit_fires():
    """The whole point of staged deployment: when exit 0 fires, later stages'
    modules are never executed (the eager wrapper claims this via exception
    unwinding; the staged runtime must deliver it structurally)."""
    torch.manual_seed(0)
    counted_late = _CountingModule(nn.Conv2d(8, 16, 3, padding=1))
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),  # 0: exit 0 attaches here
        counted_late,  # 1: must not run on early exit
        nn.AdaptiveAvgPool2d(1),  # 2
        nn.Flatten(),  # 3
        nn.Linear(16, 10),  # 4
    )
    wrapper = custom_ee(
        backbone,
        exit_layers=["0"],
        num_classes=10,
        input_shape=(1, 3, 16, 16),
        confidence_thresholds=[0.0],  # always fire at exit 0
    )
    staged = staged_model(wrapper)  # build-time probe runs the model
    staged.eval()
    counted_late.calls = 0

    result = staged.infer(torch.randn(1, 3, 16, 16))
    assert result.exit_taken == 0
    assert counted_late.calls == 0, "a fired exit must prevent later stages from running"

    # sanity: with the exit disabled the late stage does run
    wrapper.config.enabled_exits = [False]
    staged.infer(torch.randn(1, 3, 16, 16))
    assert counted_late.calls == 1


# ---------------- G. dict-output custom backbones ----------------


class _DictBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.conv(x)
        return {"features": f, "aux": f.sum()}


class _DictOutputBackbone(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.block = _DictBlock()
        self.head = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.block(x)["features"]
        return self.head(f.mean(dim=(2, 3)))


def test_custom_ee_dict_output_via_feature_extractor():
    model = custom_ee(
        _DictOutputBackbone(),
        exit_layers=["block"],
        num_classes=10,
        input_shape=(1, 3, 16, 16),
        feature_extractors={"block": lambda out: out["features"]},
    )
    model.eval()
    model.config.confidence_thresholds = [0.0]
    result = model(torch.randn(1, 3, 16, 16), mode="inference")
    assert result.exit_taken == 0
    assert result.prediction.shape == (1, 10)


# ---------------- F. genuine prior-version checkpoint migration ----------------


def test_real_v1_fixture_checkpoint_migrates():
    """tests/fixtures/v1_cifar_resnet20.pth was written by the ACTUAL
    pre-hardening save_wrapper (git main), not a hand-built dict: no
    format_version, scalar temperature 1.6, and the v1 disabled sentinel
    (confidence threshold 1.0) at exit 1. Loading must migrate all of it
    deterministically, and re-saving must produce a v2 file that round-trips.
    """
    import warnings as _warnings
    from pathlib import Path

    from earlyon.utils import load_wrapper, save_wrapper

    fixture = Path(__file__).parent / "fixtures" / "v1_cifar_resnet20.pth"
    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        model = load_wrapper(fixture)

    assert model.config.backbone == "cifar_resnet20"
    assert model.config.confidence_thresholds == [0.7, 1.0, 0.85]
    # sentinel at exit 1 became an explicit disable; the others stay enabled
    assert model.config.enabled_exits == [True, False, True]
    # scalar temperature broadcast to every head
    assert model.config.temperatures == {"e0": 1.6, "e1": 1.6, "e2": 1.6, "final": 1.6}
    assert any("enabled_exits" in str(w.message) for w in captured)

    # migrated model routes without error
    model.eval()
    result = model(torch.randn(1, 3, 32, 32), mode="inference")
    assert result.prediction.shape == (1, 10)

    # re-save writes v2 and round-trips silently (no migration warnings)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v2.pth"
        save_wrapper(model, p)
        payload = torch.load(p, map_location="cpu", weights_only=True)
        assert payload["format_version"] == 2
        with _warnings.catch_warnings(record=True) as captured2:
            _warnings.simplefilter("always")
            reloaded = load_wrapper(p)
        assert reloaded.config.enabled_exits == [True, False, True]
        assert reloaded.config.temperatures == model.config.temperatures
        migration_warnings = [w for w in captured2 if "migrated" in str(w.message)]
        assert not migration_warnings
